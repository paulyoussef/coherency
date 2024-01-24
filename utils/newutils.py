import re 
import json
import numpy as np
import copy
import torch
from transformers import BertForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader, TensorDataset, Dataset
from collections.abc import Iterable
from transformers import set_seed
import os
# get_exclude_dicts, predict_bert_batch, get_to_exclude, predict_t5, predict_bert, clean_preds_t5
def get_exclude_dicts(df, subject, object):
    # returns a mapping from sub -> all matching objs, and from obj -> all matching subjs
    # the answer to be predicted should be removed from the exclusion list                  
    exclude_dict_sub = {}
    exclude_dict_obj = {}

    df_group_sub = df.groupby(subject)
    df_group_obj = df.groupby(object)
    
    for g in df_group_sub.groups:
        exclude_dict_sub[g] = list(set(df_group_sub.get_group(g)['obj'].values))
    
    exclude_dict_obj = {}
    for g in df_group_obj.groups:
        exclude_dict_obj[g] = list(set(df_group_obj.get_group(g)['sub'].values))

    return exclude_dict_sub, exclude_dict_obj


def predict_bert(model, tokenizer, device, inputs, excluded_ids = None):

    #assert(len(excluded_ids) == len(inputs))
    
    # tokenizer inputs
    input_tokenized = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    # input ids
    input_ids = input_tokenized.input_ids
    # attention masks
    attention_masks = input_tokenized.attention_mask

    # data loading and batching
   
    dataset = TensorDataset(input_ids, attention_masks)   
    bs = 32
    data_loader = DataLoader(dataset, batch_size = bs, shuffle= False)
    model.eval()
    preds = []

    
    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            logits = model(input_ids = batch[0], attention_mask = batch[1]).logits
        # mask indices
        mask_token_indices = (batch[0] == tokenizer.mask_token_id).nonzero()[:,1]
        logits_mask = logits[torch.arange(mask_token_indices.size(0)), mask_token_indices]

        if excluded_ids != None:
            excluded_indices = excluded_ids[i*bs : i*bs + len(batch[0])]
            
            for row_idx, row_indices in enumerate(excluded_indices):
                if len(row_indices) > 0:
                    logits_mask[row_idx].index_fill_(-1, torch.tensor(row_indices).to(device), float('-inf'))
                    
        # predictions
        predicted_tokens_ids = logits_mask.argmax(axis=-1)
        preds += tokenizer.batch_decode(predicted_tokens_ids)
        
    return preds

def get_to_exclude(tokenizer, exclude_dict, preds, corrects, relation_type):
    # collect excluded_ids, excluded words
    excluded_ids = []
    excluded_words = []
    # go through predictions from previous step and correct answers 
    for p, c in zip(preds, corrects): 
        tmp = exclude_dict.get(p, [])
    
        words = tmp.copy() 
        
        # return excluded ids
        if len(words) == 0:
            excluded_ids.append([])
            excluded_words.append(None)
            continue
            
        else:
            # ids of excluded words 
            ids = set(list(flatten([tokenizer(x, add_special_tokens=False).input_ids for x in words])))
            correct_id = set(list(flatten(tokenizer(c, add_special_tokens=False).input_ids)))
            
            # make sure correct answer is not excluded
            ids = (list(ids.difference(correct_id)))
            # could contain subwords (TODO how to handle this later)
            words_decoded = [tokenizer.decode(x) for x in ids]

            assert (len(ids) == len(words_decoded))

            if len(ids) == 0:
                excluded_ids.append([])
                excluded_words.append(None)
            else: 
                excluded_ids.append(ids)
                excluded_words.append(words_decoded)

    if 't5' in str(type(tokenizer)): 
        #print('before: ', excluded_ids)
        ids_t5 =[]
        for i in excluded_ids: 
            if len(i) == 0:
                ids_t5.append(None)
            else:
                ids_t5.append([[j] for j in i])
                    
        #print('after: ', ids_t5)
        return ids_t5, excluded_words

    return excluded_ids, excluded_words


def predict_t5(model, tokenizer, device, inputs, batch_size = 32, excluded_ids = None):

    #assert(len(excluded_ids) == len(inputs))
    
    # tokenizer inputs
    input_tokenized = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    # input ids
    input_ids = input_tokenized.input_ids
    # attention masks
    attention_masks = input_tokenized.attention_mask
    
    # data loading and batching
   
    dataset = TensorDataset(input_ids, attention_masks)   
    bs = batch_size
    data_loader = DataLoader(dataset, batch_size = bs, shuffle= False)
    model.eval()
    preds = []

    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            outputs = model.generate(input_ids = batch[0], attention_mask = batch[1], 
                                     num_beams=10, num_return_sequences=1, max_length=50, bad_words_ids = excluded_ids)
        preds += tokenizer.batch_decode(outputs)
        
    return preds




def predict_batch(model, tokenizer, device, inputs, excluded_ids = None):
    if 'bert' in str(type(model)).lower():
        return predict_bert_batch(model, tokenizer, device, inputs, excluded_ids)
    elif 't5' in str(type(model)).lower():
        assert excluded_ids == None
        return predict_t5_batch(model, tokenizer, device, inputs)
    else:
        raise ValueError



def clean_preds_t5(preds):
    pattern = r'<extra_id_0>(.*?)<'

    preds_clean = []
    for sent in preds: 
        matches = re.findall(pattern, sent)
        
        if len(matches) == 0:
            pred = sent[17:].split('.')[0].strip()
            print('noutput: {}\npred: {}'.format(sent, pred))
            preds_clean.append(pred)
        else:
            pred = matches[0].strip()
            if len(pred) > 1 and pred.endswith('.'):
                pred =  pred[:-1]
            preds_clean.append(pred)
            
    return preds_clean

def clean_preds_ssm(preds):
    pattern = r'<pad>(.*?)<'

    preds_clean = []
    for sent in preds: 
        matches = re.findall(pattern, sent)
        
        if len(matches) == 0:
            pred = sent[5:].split('.')[0].strip()
            print('noutput: {}\npred: {}'.format(sent, pred))
            preds_clean.append(pred)
        else:
            pred = matches[0].strip()
            if len(pred) > 1 and pred.endswith('.'):
                pred =  pred[:-1]
            preds_clean.append(pred)
            
    return preds_clean
    
def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def get_template(relation, filename='../data/lama/relations.jsonl'):
    relations = load_jsonl(filename)
    for r in relations: 
        if r['relation'].lower() == relation.lower():
            type = None #r['type']
            return r['template'], None
    raise ValueError


def get_ar_templates(relation, filename='../data/lama/relations_autoregressive.jsonl'):
    relations = load_jsonl(filename)
    for r in relations: 
        if r['relation'].lower() == relation.lower():
            type = None #r['type']
            return r['template_obj'], r['template_sub']
    raise ValueError


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


## For diverse

def predict_t5_diverse(model, tokenizer, device, inputs, batch_size = 32, excluded_ids = None, seed=42):

    #assert(len(excluded_ids) == len(inputs))
    
    # tokenizer inputs
    input_tokenized = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    # input ids
    input_ids = input_tokenized.input_ids
    # attention masks
    attention_masks = input_tokenized.attention_mask
    
    # data loading and batching
   
    dataset = TensorDataset(input_ids, attention_masks)   
    bs = batch_size
    num_topk_gens = 10
    data_loader = DataLoader(dataset, batch_size = bs, shuffle= False)
    model.eval()
    preds = []

    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            outputs_beam = model.generate(input_ids = batch[0], attention_mask = batch[1], 
                                     num_beams=10, num_return_sequences=1, max_length=50, bad_words_ids = excluded_ids)
            set_seed(seed)
            outputs_topk = model.generate(input_ids = batch[0], attention_mask = batch[1], 
                                    do_sample=True, top_k= num_topk_gens, max_length=50, 
                                          num_return_sequences = num_topk_gens, bad_words_ids = excluded_ids)

            
            
            preds_beam = tokenizer.batch_decode(outputs_beam)
            preds_diverse = tokenizer.batch_decode(outputs_topk)
            # convert diverse preds to equally sized chunks of size 'num_topk_gens'
            if batch_size > 1: 
                preds_diverse = [preds_diverse[i:i+num_topk_gens] for i in range(0, len(preds_diverse), 10)]
                aligned_preds = [[beam] + diverse for beam, diverse in zip(preds_beam, preds_diverse)]
            else:
                aligned_preds = preds_beam + preds_diverse
            
            
        preds += aligned_preds
        
    return preds

def predict_t5_diverse_v2(model, tokenizer, device, inputs, batch_size = 1, excluded_ids = None, seed=42):
    # improved version of 'predict_t5_diverse'. Past generations are excluded. Only beam seach is used
    input_tokenized = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    # input ids
    input_ids = input_tokenized.input_ids
    # attention masks
    attention_masks = input_tokenized.attention_mask
    
    # data loading and batching
   
    dataset = TensorDataset(input_ids, attention_masks)   
    bs = batch_size
    num_generations = 5
    data_loader = DataLoader(dataset, batch_size = bs, shuffle= False)
    model.eval()
    preds = []

    for i, batch in enumerate(data_loader):
        aligned_preds = []
        # create a copy of excluded_ids for each instance 
        current_exluded_ids = copy.deepcopy(excluded_ids)
        with torch.no_grad():
            for _ in range(num_generations):
                outputs_beam = model.generate(input_ids = batch[0], attention_mask = batch[1], 
                                     num_beams=10, num_return_sequences=1, max_length=50, bad_words_ids = current_exluded_ids)

                # Find the indices of 32099 <extra_id_0> and 32098 <extra_id_1>
                start_idx = (outputs_beam == 32099).nonzero()[0, 1] if (outputs_beam == 32099).any() else 0
                end_idx = (outputs_beam == 32098).nonzero()[0, 1] if (outputs_beam == 32098).any() else -1

                if end_idx >= start_idx:
                    current_pred = outputs_beam[0, start_idx + 1:end_idx].cpu().numpy()
    
                    if current_exluded_ids == None:
                        current_exluded_ids = []

                    curr_pred = list(set(current_pred) - set([ 0, 32099, 32098]))
                    for tmp in current_pred:
                        current_exluded_ids.append([tmp])
                
                preds_beam = tokenizer.batch_decode(outputs_beam)[0]               
                aligned_preds.append(preds_beam)
            
            
        preds += [aligned_preds]
        
    return preds
    
def clean_preds_t5_diverse(ls):

    clean_preds = []
    
    for l in ls:
        clean_preds.append(clean_preds_t5(l))
        
    return clean_preds


def get_ent_proba(model, tokenizer, device, input, ent):
    #print('input: ', input, ' ent: ', ent)
    # input: input with MASK
    # ent: entity 

    # ids corresponding to entities
    ent_ids = tokenizer(ent, add_special_tokens = False).input_ids
    # entity parts (tokens)
    ent_tokens = tokenizer.tokenize(ent)
    # probabilities for each part 
    probs = []
    #print('input: ', input)
    # a copy of the entity
    tmp = tokenizer.mask_token*len(ent_tokens)
    for ent_token in ent_tokens:
        tmp.replace(ent_token.replace('#', ''), tokenizer.mask_token)
    
    input = input.replace('<extra_id_0>', tmp)
    #print('maske_input: ', input)
    input_tokenized = tokenizer(input, return_tensors='pt')
        #print(input_tokenized)
    input_tokenized.to(device)
    
        
    # mask token index
    mask_indices = (input_tokenized.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=False)
    mask_indices = mask_indices.cpu().detach().numpy().flatten()

    model.eval()
    with torch.no_grad():
        logits = model(**input_tokenized).logits
           
    for ent_id, m in zip(ent_ids, mask_indices):
        #print(logits.shape)
        # Shape:     [1, 10, 30522]
        logits_mask = logits[0, m]
        #print(logits_mask.softmax(axis=-1)[ent_id])
        probs.append(logits_mask.softmax(axis=-1)[ent_id].cpu().detach().numpy())

    #print(np.array(probs))
    #print(ent_tokens)
    assert len(probs) == len(ent_tokens)
    return np.mean(np.array(probs))
    

def get_most_probable_entity(backup_model, backup_tokenizer, device, input, answers):
    max_index = -1
    max_proba = -1

    counter = 0
    for ent in answers:
        proba = get_ent_proba(backup_model, backup_tokenizer, device, input, ent)
        if proba > max_proba:
            max_proba = proba
            max_index = counter    
        counter += 1

    return answers[max_index]


def get_backup_preds(backup_model, backup_tokenizer, device, inputs, preds):
    backup_preds = []

    assert (len(inputs) == len(preds))
    for i, answers in zip(inputs, preds):
        backup_pred = get_most_probable_entity(backup_model, backup_tokenizer, device, i, answers)
        # print('backup pred: ', backup_pred)
        backup_preds.append(backup_pred)

    return backup_preds

def clean_preds_ssm_diverse(ls):

    clean_preds = []
    
    for l in ls:
        clean_preds.append(clean_preds_ssm(l))
        
    return clean_preds


def partial_match(s1, s2):
    if s1.strip() == '' or s2.strip() == '':
        return False
    if s1.strip().lower() in s2.strip().lower():
        return True
    elif s2.strip().lower() in s1.strip().lower():
        return True
    else:
        return False

def is_x_before_y(input_string):
    assert "[X]" in input_string
    assert "[Y]" in input_string
    # Find the positions of [X] and [Y] in the string
    x_position = input_string.find("[X]")
    y_position = input_string.find("[Y]")

    # Check if both [X] and [Y] are present in the string
    if x_position != -1 and y_position != -1:
        return x_position < y_position
    else:
        return False  # One or both placeholders are missing in the string

def get_paraphrased_templates(relation):
    path = './data/pararel/{}.jsonl'.format(relation)
    if not os.path.exists(path):
      return []
    ls = load_jsonl(path)
    templates = []
    for l in ls:
        templates.append(l['pattern'])
    return templates

class TypedDataset(Dataset):
    def __init__(self, all_inputs):
        # e.g., The capital of Germany is..
        self.inputs = [x[0] for x in all_inputs]
        # e.g., Berlin
        self.answers = [x[1] for x in all_inputs]


    def __getitem__(self, index):
        i = self.inputs[index]
        a = self.answers[index]
        return i, a #, ix
    
    def __len__(self):
        return len(self.answers)

def predict_typed_preds(df, model, tokenizer, device, inputs_column, batch_size=128):
    outputs_column = inputs_column.replace('inputs', 'pred')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # aggregate over all instances
    all_entities_probs = []

    for _, row in df.iterrows():
        # Entities probas for the current row (e.g., [(Berlin, 0.5),..., (Paris, 0.3)])
        ents_probas = []
        # All possible inputs for this tuple
        all_inputs = row[inputs_column]
        # collect all entities and their probabilities for all instances
        # This dataset corresponds to one tuple
        dataset = TypedDataset(all_inputs=all_inputs)

        data_loader = DataLoader(dataset, batch_size = batch_size, shuffle= False)
        
        for batch in data_loader:
            inputs = tokenizer(list(batch[0]), return_tensors="pt", padding=True, truncation=True).to(device)
            labels = tokenizer(list(batch[1]), return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)

            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'], pad_token_id=tokenizer.eos_token_id, return_dict_in_generate = True, output_scores = True, min_new_tokens= labels['input_ids'].shape[1], max_new_tokens= labels['input_ids'].shape[1])
                # Shape: batch size x labels length x vocab size
            output_probas = torch.stack(list(outputs.scores), dim=1).log_softmax(-1)
            selected_probabilities = output_probas.gather(dim=2, index=labels["input_ids"].unsqueeze(2)).squeeze(-1) # shape: num_instances x num_instaces in output labels 
            # zero out padded tokens 
            selected_probabilities = torch.where(selected_probabilities == float('-inf'), torch.tensor(0.0), selected_probabilities)

            selected_probabilities = selected_probabilities * labels['attention_mask']

            final_probas = selected_probabilities.sum(axis=-1) / selected_probabilities.count_nonzero(dim=-1)
            #avg_probas = list(selected_probabilities.mean(axis=-1).cpu().numpy())    
            avg_probas = final_probas.cpu().numpy()
            ents_probas += [(x,y) for x,y in zip(batch[1], avg_probas)]
        all_entities_probs.append(ents_probas)
    df[outputs_column + '_entities_probs'] = all_entities_probs
    df[outputs_column + '_entities_probs'] = df.apply(lambda x: sorted(x[outputs_column + '_entities_probs'], key= lambda y: y[1], reverse=True), axis=1 )
    df[outputs_column] = df.apply(lambda x : x[outputs_column + '_entities_probs'][0][0], axis=1 )
    
    return df

def predict_typed_preds_optimized(df, model, tokenizer, device, inputs_column, batch_size=128):
    outputs_column = inputs_column.replace('inputs', 'pred')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # aggregate over all instances
    all_entities_probs = []

    for _, row in df.iterrows():
        # Entities probas for the current row (e.g., [(Berlin, 0.5),..., (Paris, 0.3)])
        ents_probas = []
        # All possible inputs for this tuple
        all_inputs = row[inputs_column]
        # collect all entities and their probabilities for all instances
        # This dataset corresponds to one tuple
        dataset = TypedDataset(all_inputs=all_inputs)
        previous_outputs_shape = None

        data_loader = DataLoader(dataset, batch_size = batch_size, shuffle= False)
        model.eval()
        for batch in data_loader:
            labels = tokenizer(list(batch[1]), return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)

            current_outputs_shape = labels['input_ids'].shape

            if current_outputs_shape != previous_outputs_shape:
                inputs = tokenizer(list(batch[0]), return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    outputs = model.generate(inputs['input_ids'], pad_token_id=tokenizer.eos_token_id, return_dict_in_generate = True, output_scores = True, min_new_tokens= labels['input_ids'].shape[1],  max_new_tokens= labels['input_ids'].shape[1])
                # Shape: batch size x labels length x vocab size
                    
                output_probas = torch.stack(list(outputs.scores), dim=1).log_softmax(-1)
            
            selected_probabilities = torch.clone(output_probas).gather(dim=2, index=labels["input_ids"].unsqueeze(2)).squeeze(-1) # shape: num_instances x num_instaces in output labels 
            selected_probabilities = torch.where(selected_probabilities == float('-inf'), torch.tensor(0.0), selected_probabilities)

            # zero out padded tokens 
            selected_probabilities = selected_probabilities * labels['attention_mask']
            final_probas = selected_probabilities.sum(axis=-1) / selected_probabilities.count_nonzero(dim=-1)
            #avg_probas = list(selected_probabilities.mean(axis=-1).cpu().numpy())    
            avg_probas = final_probas.cpu().numpy()
            ents_probas += [(x,y) for x,y in zip(batch[1], avg_probas)]
            previous_outputs_shape = labels['input_ids'].shape
        all_entities_probs.append(ents_probas)
    df[outputs_column + '_entities_probs'] = all_entities_probs
    df[outputs_column + '_entities_probs'] = df.apply(lambda x: sorted(x[outputs_column + '_entities_probs'], key= lambda y: y[1], reverse=True), axis=1 )
    df[outputs_column] = df.apply(lambda x : x[outputs_column + '_entities_probs'][0][0], axis=1 )

    return df