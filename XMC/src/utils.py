import os
import math
import datetime
import numpy as np
import torch
import random


class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, text, visual=True):
        if visual:
            print(text)
        with open(f'./log/{self.name}', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')

def get_exp_name(dataset,bert,num_group):
    name = [dataset, '' if bert == 'bert-base' else bert]
    if dataset in ['wiki500k', 'amazon670k', 'amazon13m']:
        name.append(str(num_group))

    return '_'.join([i for i in name if i != ''])

def init_seed(seed):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('log'):
        os.makedirs('log')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def check_gorup(n_labels,num_group):
    num_ele_per_group = math.ceil(n_labels / num_group)
    if (num_group * num_ele_per_group - n_labels) >= num_ele_per_group:
        return check_gorup(n_labels,num_group-1)
    else:
        return [num_group, num_ele_per_group]

def get_groups(n_labels, num_group, num_ele_per_group):
    group_y = []
    bs = 0
    for i in range(num_group-1):
        group_y.append(np.array([bs + _ for _ in range(num_ele_per_group)]))
        bs += num_ele_per_group
    if n_labels - bs>0:
        group_y.append(np.array([bs+_ for _ in range(n_labels -bs)]))
    assert len(group_y) == num_group
    return np.array(group_y, dtype=None if len(group_y[-1])==len(group_y[-2]) else object)

def get_groups_v2(n_labels, num_group, num_ele_per_group):
    group_y , idx = [] , 0
    for i in range(num_group-1):
        group_y.append(np.array([idx + _ for _ in range(num_ele_per_group)]))
        idx += num_ele_per_group
    has_padding = False
    if n_labels - idx>0:
        last_y = np.array([idx + _ for _ in range(n_labels -idx)])
        if len(last_y)< num_ele_per_group:
            num_pad = num_ele_per_group - len(last_y)
            #last_y = np.pad(last_y,(0, num_pad), 'constant', constant_values=n_labels)
            has_padding = True
            print('*** pad last group with constant mode, \
                  length = ',num_pad)
            print(group_y[-1])
            print(last_y)
        group_y.append(last_y)
    #assert len(group_y) == num_group and len(group_y[-1])==num_ele_per_group
    #return np.array(group_y,dtype=object)#, has_padding
    return np.array(group_y, dtype=None if len(group_y[-1])==len(group_y[-2]) else object)
