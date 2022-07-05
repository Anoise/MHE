import os
import torch
import pickle
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

import tqdm

def createDataCSV(dataset,path='/home/user/Data'):
    labels = []
    texts = []
    dataType = []
    label_map = {}

    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K',
                'amazon3m':'Amazon-3M'}

    assert dataset in name_map
    dataset = name_map[dataset]
    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    with open(path+f'/data/{dataset}/train{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')

    with open(path+f'/data/{dataset}/test{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')

    with open(path+f'/data/{dataset}/train_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))


    with open(path+f'/data/{dataset}/test_labels.txt') as f:
        print(len(label_map))
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))
        print(len(label_map))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)

    print('label map', len(label_map))

    return df, label_map


class MDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df, self.n_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        # if mode=='test':
        #     self.df = self.df[:int(0.15*len(self.df))]
           
        self.len = len(self.df)
        self.tokenizer, self.max_length, self.group_y = tokenizer, max_length, group_y
        self.multi_group = False
        self.token_type_ids = token_type_ids
        self.candidates_num = candidates_num
        self.max_lbl = 100

        if group_y is not None:
            # group y mode
            self.candidates_num, self.n_group_y_labels = candidates_num, group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.long)
            for idx, labels in enumerate(group_y):
                self.map_group_y[labels] = idx
          
    def __getitem__(self, idx):
        max_len = self.max_length
        review = self.df.text.values[idx].lower()
        labels = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]
        review = ' '.join(review.split()[:max_len])
        text = review
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
        else:
            # fast 
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length-1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        if self.group_y is not None:
            group_labels = self.map_group_y[labels]
            if self.multi_group:
                group_labels = np.concatenate(group_labels)
            group_label_ids = torch.zeros(self.n_group_y_labels)
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),
                                      torch.tensor([1.0 for i in group_labels]))
  
            _labels = labels + [-1 for _ in range(self.max_lbl - len(labels))] \
                    if len(labels)<self.max_lbl else labels[:self.max_lbl]

            if self.mode == 'train':
                return input_ids, attention_mask, token_type_ids,\
                    torch.LongTensor(_labels), group_label_ids, 1
            else:
                return input_ids, attention_mask, token_type_ids,\
                    torch.LongTensor(_labels), group_label_ids, 1
        
        label_ids = torch.zeros(self.n_labels)
        label_ids = label_ids.scatter(0, torch.tensor(labels),
                                      torch.tensor([1.0 for i in labels]))

        return input_ids, attention_mask, token_type_ids, label_ids
  
    
    def __len__(self):
        return self.len 
