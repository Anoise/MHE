import tqdm
import torch
import pandas as pd
import numpy as np

def createDataCSV(dataset, path):
    labels = []
    texts = []
    dataType = []
    label_map = {}

    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K'}

    assert dataset in name_map
    dataset = name_map[dataset]
    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    with open(path+f'/{dataset}/train{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')

    with open(path+f'/{dataset}/test{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')

    with open(path+f'/{dataset}/train_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))


    with open(path+f'/{dataset}/test_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)

    return df, label_map


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 num_classes=None,token_type_ids=None,max_label=60):

        self.df = df[df.dataType == mode]
        self.n_labels = len(label_map)
        self.len = len(self.df)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        self.num_classes = num_classes
        self.token_type_ids = token_type_ids
        self.max_lbl = max_label
        print('model ',mode,' num class',num_classes if num_classes else 'no group ...')
        
    def __getitem__(self, idx):
        review = self.df.text.values[idx].lower()
        review = ' '.join(review.split()[:self.max_length])
        _labels = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]
        labels = _labels + [-1 for _ in range(self.max_lbl - len(_labels))] \
                    if len(_labels)<self.max_lbl else _labels[:self.max_lbl]
        input_ids, attention_mask, token_type_ids = self.get_tonkers(review, idx)
        if self.num_classes:
            _labels = torch.tensor(np.array(_labels) // self.num_classes[1])
            group_labels = torch.zeros(self.num_classes[0])
            group_labels = group_labels.scatter(0, _labels, torch.tensor([1.0 for _ in _labels]))
            return input_ids, attention_mask, token_type_ids, torch.LongTensor(labels), group_labels
        return input_ids, attention_mask, token_type_ids, torch.LongTensor(labels)

    def get_tonkers(self, text, idx):
        """ getting input_ids, attention_mask, token_type_ids for pretraining nlp model 
        Args:
            text (str): input text 
            idx (int): index item

        Returns:
            tuple: input_ids, attention_mask, token_type_ids
        """
        #  input_ids ....
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True
            )
        else:
            # fast 
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length-1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]
        
        # attention_mask & token_type_ids    
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)
        
        return input_ids, attention_mask, token_type_ids
    
    def __len__(self):
        return self.len 
