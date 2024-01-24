from transformers import BertForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, RobertaForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
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
from newutils import get_exclude_dicts, get_to_exclude, predict_t5, predict_bert, clean_preds_t5, clean_preds_ssm, get_ar_templates, load_jsonl, flatten, partial_match, predict_typed_preds_optimized


def prepare_inputs_obj(tokenizer, template, sub, objects):
    assert (template.endswith('[Y]'))
    model_temp = '{}'
    res = []
    for o in objects:  
        res.append((template.replace('[X]', sub).replace('[Y]', ''), model_temp.format(o)))
    return res 

def prepare_inputs_sub(tokenizer, template, obj, subjects):
    assert (template.endswith('[X]'))
    model_temp = '{}'
    res = []
    for s in subjects:  
        res.append((template.replace('[Y]', obj).replace('[X]', ''), model_temp.format(s)))
    return res 

def prepare_inputs_sub2(tokenizer, template, obj, subjects, to_exclude, gt_sub):
    assert (template.endswith('[X]'))
    # subjects that need to be excluded
    subjects_local = subjects.copy()

    for x in to_exclude: 
        if x != gt_sub:
            subjects_local.remove(x)

    model_temp = '{}'
    res = []
    assert (gt_sub in subjects_local)
    for s in subjects_local:  
        res.append((template.replace('[Y]', obj).replace('[X]', ''), model_temp.format(s)))
    return res 


def prepare_inputs_obj2(tokenizer, template, sub, objects, to_exclude, gt_obj):
    assert (template.endswith('[Y]'))
    # subjects that need to be excluded
    objects_local = objects.copy()

    for x in to_exclude: 
        if x != gt_obj:
            objects_local.remove(x)

    model_temp = '{}'
    res = []
    assert (gt_obj in objects_local)
    for s in objects_local:  
        res.append((template.replace('[X]', sub).replace('[Y]', ''), model_temp.format(s)))
    return res 

def run(model_name, relation_name, batch_size):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    MODEL_NAME = model_name
    RELATION = relation_name
    BS = batch_size
    
    if 'gpt2' in MODEL_NAME.lower():
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif 'gpt-neo' in MODEL_NAME.lower():
        model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    else:
        raise ValueError

    model.to(device)
    #RELATION = 'P36'

    subject = 'sub'
    object = 'obj'
    
    relation = load_jsonl('./data/lama/{}.jsonl'.format(RELATION))
    
    temp_obj, temp_sub = get_ar_templates(RELATION, filename='./data/lama/relations_autregressive.jsonl')
    
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
    

    
    # to fiterout easy examples
    df = df[df.apply(lambda x: (x[subject] not in x[object]) and (x[object] not in x[subject]), axis=1)]


    if len(df) == 0:
        print('No data for this relation!')
        with open('results_ar_lms.csv', 'a') as f:
            f.write('{}, {}, {}, {}, {}\n'.format(MODEL_NAME, RELATION, len(df), 'NA', 'NA'))

        return 
    print(df)
    
    # TODO: augment objects to avoid to easy instances

    # would predict objects
    df['inputs11'] = df.apply(lambda x: prepare_inputs_obj(tokenizer, temp_obj, x[subject], list(set(objs))), axis=1)
    # would predict subjects
    df['inputs21'] = df.apply(lambda x: prepare_inputs_sub(tokenizer, temp_sub, x[object], list(set(subs))), axis=1)

    # predicting objects
    # [X] some template [MASK]  --> [Y] == pred11 / predicting objects
    predict_typed_preds_optimized(df, model, tokenizer, device, 'inputs11', batch_size=batch_size)
    # predicting subjects
    predict_typed_preds_optimized(df, model, tokenizer, device, 'inputs21', batch_size=batch_size)

    # exclude_dict_obj mapping from [Y] to all [X]'s
    # exclude_dict_sub mapping from [X] to all [Y]'s
    exclude_dict_sub, exclude_dict_obj = get_exclude_dicts(df, subject, object)


    # Round 2
    # need mapping from objects to subjects (exclude_dict_obj)
    df['excluded12'] = df.apply(lambda x:  exclude_dict_obj.get(x['pred11'], []), axis=1)
    df['excluded22'] = df.apply(lambda x:  exclude_dict_sub.get(x['pred21'], []), axis=1)

    df['inputs12'] = df.apply(lambda x: prepare_inputs_sub2(tokenizer, temp_sub, x['pred11'], list(set(subs)), x['excluded12'], x[subject]), axis=1)
    df['inputs22'] = df.apply(lambda x: prepare_inputs_obj2(tokenizer, temp_obj, x['pred21'], list(set(objs)), x['excluded22'], x[object]), axis=1)

    # predicting subjects
    predict_typed_preds_optimized(df, model, tokenizer, device, 'inputs12', batch_size=batch_size)
    # predicting objects
    predict_typed_preds_optimized(df, model, tokenizer, device, 'inputs22', batch_size=batch_size)

    print('Sample from inputs12:')
    print(df['inputs12'].head())
    print('Sample from inputs21:')
    print(df['inputs21'].head())



    scores1 = []
    scores2 = []
    for i, row in df.iterrows():
        if  partial_match(row['pred12'], row[subject]) and row['pred12'].strip().lower() != row['pred11'].strip().lower():
            scores1.append(1)
        else:
            scores1.append(0)
            
        if partial_match(row['pred22'], row[object]) and row['pred22'].strip().lower() != row['pred21'].strip().lower():
            scores2.append(1)
        else:
            scores2.append(0)
    
    
    df['score1'] = scores1
    df['score2'] = scores2
    

    df.to_csv('./results_ar_lms/results_{}_{}.csv'.format(MODEL_NAME.replace('/', '_'), RELATION), index=False)
    
    with open('results_ar_lms.csv', 'a') as f:
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

