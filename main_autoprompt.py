from transformers import BertForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, RobertaForMaskedLM, BertTokenizer
import torch
import torch.nn as nn
import pandas as pd
import ast
import tqdm as notebook_tqdm
import os 
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import argparse

sys.path.append('./utils/')
from newutils import get_exclude_dicts, get_to_exclude, predict_t5, predict_bert, clean_preds_t5, clean_preds_ssm, get_template, load_jsonl, flatten, partial_match

def run(model_name, relation_name, batch_size):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    MODEL_NAME = model_name
    RELATION = relation_name
    BS = batch_size
    
    if 'bert-base' in MODEL_NAME.lower() or 'bert-large' in MODEL_NAME.lower():
        model = BertForMaskedLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        MASK_TOKEN = tokenizer.mask_token
        
    elif 'informbert' in MODEL_NAME.lower():
        # This model uses the roberta class, and bert tokenizer 
        model = RobertaForMaskedLM.from_pretrained(MODEL_NAME)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        MASK_TOKEN = tokenizer.mask_token
        
    elif 't5' in MODEL_NAME.lower():
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, legacy = False)
        MASK_TOKEN = tokenizer.additional_special_tokens[0] # '<extra_id_0>'
    else:
        raise ValueError

    model.to(device)
    #RELATION = 'P36'

    subject = 'sub'
    object = 'obj'
    
    relation = load_jsonl('./data/lama/{}.jsonl'.format(RELATION))
    
    template, relation_type = get_template(RELATION,  filename='./data/autoprompt/fact_retrieval_bert_prompts.jsonl')
    input_text = template.replace('[X]', '{}').replace('[Y]', '{}')
    
    subs = []
    objs = []
    
    for i in relation:
        if i['sub_label'] != i['obj_label']:
            subs.append(i['sub_label'])
            objs.append(i['obj_label'])
            
    df = pd.DataFrame(
        {
            subject : subs,
            object : objs
        }
    )
    
    
    # filter out multi-token entities for BERT
    if 'bert' in MODEL_NAME.lower():
        df = df[df.apply(lambda x: (len(tokenizer.tokenize(x[subject])) == 1) and (len(tokenizer.tokenize(x[object])) == 1), axis=1)]
    
    # convert to lower case if uncased
    if 'uncased' in MODEL_NAME.lower():
        df[subject] = df[subject].apply(lambda x: x.lower())
        df[object] = df[object].apply(lambda x: x.lower())
    
    # to fiterout easy examples
    df = df[df.apply(lambda x: (x[subject] not in x[object]) and (x[object] not in x[subject]), axis=1)]
    if len(df) == 0:
        print('No data for this relation!')
        with open('results_autoprompt.csv', 'a') as f:
            f.write('{}, {}, {}, {}, {}\n'.format(MODEL_NAME, RELATION, len(df), 'NA', 'NA'))

        return 
    print(df)
    
    inputs11 = df[subject].apply(lambda x: input_text.format(x, MASK_TOKEN)).tolist()
    print('Sample from inputs11:')
    print(inputs11[:5])

    inputs21 = df[object].apply(lambda x: input_text.format(MASK_TOKEN, x)).tolist()
    print('Sample from inputs21:')
    print(inputs21[:5])

    # Round 1
    if 'bert' in MODEL_NAME.lower():
        preds11 = predict_bert(model, tokenizer, device, inputs11, excluded_ids = None)
        preds21 = predict_bert(model, tokenizer, device, inputs21, excluded_ids = None)

        if 'informbert' in MODEL_NAME.lower():
            preds11 = [x.replace(' ', '') for x in preds11 ]
            preds21 = [x.replace(' ', '') for x in preds21 ]

    elif 't5' in MODEL_NAME.lower():
        preds11_raw = predict_t5(model, tokenizer, device, inputs11, batch_size = BS, excluded_ids = None)
        preds21_raw = predict_t5(model, tokenizer, device, inputs21, batch_size = BS, excluded_ids = None)

        if 'ssm' in MODEL_NAME.lower():
            preds11 = clean_preds_ssm(flatten(preds11_raw))
            preds21 = clean_preds_ssm(flatten(preds21_raw))
        else: 
            preds11 = clean_preds_t5(preds11_raw)
            preds21 = clean_preds_t5(preds21_raw)
    else:
        raise ValueError
    # Round 2
    # prepare exclusion dicts
    exclude_dict_sub, exclude_dict_obj = get_exclude_dicts(df, subject, object)

    inputs12 = pd.Series(preds11).apply(lambda x: input_text.format(MASK_TOKEN, x)).tolist()
    inputs22 = pd.Series(preds21).apply(lambda x: input_text.format(x, MASK_TOKEN)).tolist()

    print('Sample from inputs12:')
    print(inputs12[:5])
    print('Sample from inputs22:')
    print(inputs22[:5])

    if 'bert' in MODEL_NAME.lower():
        excluded_ids12, excluded_words12 = get_to_exclude(tokenizer, exclude_dict_obj, preds11, df[subject], relation_type)
        excluded_ids22, excluded_words22 = get_to_exclude(tokenizer, exclude_dict_sub, preds21, df[object], relation_type)
        preds12 = predict_bert(model, tokenizer, device, inputs12, excluded_ids = excluded_ids12)
        preds22 = predict_bert(model, tokenizer, device, inputs22, excluded_ids = excluded_ids22)

        if 'informbert' in MODEL_NAME.lower():
            preds12 = [x.replace(' ', '') for x in preds12 ]
            preds22 = [x.replace(' ', '') for x in preds22 ]
        
    elif 't5' in MODEL_NAME.lower():
        excluded_ids12, excluded_words12 = get_to_exclude(tokenizer, exclude_dict_obj, preds11, df[subject], relation_type)
        excluded_ids22, excluded_words22 = get_to_exclude(tokenizer, exclude_dict_sub, preds21, df[object], relation_type)
        preds12_raw = [predict_t5(model, tokenizer, device, i, batch_size = 1, excluded_ids = e) for i, e in zip(inputs12, excluded_ids12)]
        preds22_raw = [predict_t5(model, tokenizer, device, i, batch_size = 1, excluded_ids = e) for i, e in zip(inputs22, excluded_ids22)]

        if 'ssm' in MODEL_NAME.lower():
            preds12 = clean_preds_ssm(flatten(preds12_raw))
            preds22 = clean_preds_ssm(flatten(preds22_raw))
        else:
            preds12 = clean_preds_t5(flatten(preds12_raw))
            preds22 = clean_preds_t5(flatten(preds22_raw))

    results = pd.DataFrame(
    {
        subject : df[subject],
        object : df[object],
        'input11': inputs11,
        'pred11': preds11,

        'input12': inputs12,
        'excluded12': excluded_words12,
        'pred12': preds12,

        'input21': inputs21, 
        'pred21': preds21,

        'input22': inputs22, 
        'excluded22': excluded_words22,
        'pred22': preds22,
    }
)

    scores1 = []
    scores2 = []
    for i, row in results.iterrows():
        if  partial_match(row['pred12'], row[subject]) and row['pred12'].strip().lower() != row['pred11'].strip().lower():
            scores1.append(1)
        else:
            scores1.append(0)
            
        if partial_match(row['pred22'], row[object]) and row['pred22'].strip().lower() != row['pred21'].strip().lower():
            scores2.append(1)
        else:
            scores2.append(0)
    
    
    results['score1'] = scores1
    results['score2'] = scores2
    
    if 't5' in MODEL_NAME.lower():
        results['preds11_raw'] = preds11_raw
        results['preds21_raw'] = preds21_raw
        results['preds12_raw'] = preds12_raw
        results['preds22_raw'] = preds22_raw

    results.to_csv('./results_autoprompt/results_autoprompt_{}_{}.csv'.format(MODEL_NAME.replace('/', '_'), RELATION), index=False)
    
    with open('results_autoprompt.csv', 'a') as f:
        f.write('{}, {}, {}, {}, {}\n'.format(MODEL_NAME, RELATION, len(df), np.mean(scores1+scores2), np.mean(np.bitwise_and(scores1, scores2))))



def main():
    parser = argparse.ArgumentParser(description="Example script to capture arguments and pass to a method")
    parser.add_argument("--model", required=True, type = str, help="e.g., 'bert-base-uncased'")
    parser.add_argument("--relation", required=True, type=str, help="relation")
    parser.add_argument("--bs", required=True, type=int, help="batch size")

    args = parser.parse_args()
    run(args.model, args.relation, args.bs)

if __name__ == '__main__':
    main()

