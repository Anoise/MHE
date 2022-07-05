import os
import math
import random
import datetime
import functools
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from timm.utils import NativeScaler
from timm.utils import dispatch_clip_grad
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer,RobertaTokenizerFast
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from tokenizers import BertWordPieceTokenizer

print('utils copy ...')

class Logger():
    def __init__(self, name='training.log'):
        self.name = name

    def __call__(self, text, visual=True):
        if visual: print(text)
        with open(f'./log/{self.name}', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')

def get_exp_name(args):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('log'):
        os.makedirs('log')
    name = [args.dataset, '' if args.model == 'bert-base' else args.model]
    return '_'.join([i for i in name if i != ''])

def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_model(model_name,path=''):
    if 'roberta' in model_name:
        print('load roberta-base')
        
        model_config = RobertaConfig.from_pretrained(os.path.join(path,'roberta-base'))
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained(os.path.join(path,'roberta-base'), config=model_config)
    elif 'xlnet' in model_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained(os.path.join(path,'xlnet-base-cased'))
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained(os.path.join(path,'xlnet-base-cased'), config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained(os.path.join(path,'bert-base-uncased'))
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained(os.path.join(path,'bert-base-uncased'), config=model_config)
    return bert

def get_optimizer_params(model, no_decay=['bias', 'LayerNorm.weight']):
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() 
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    return optimizer_grouped_parameters

def get_loss_scaler(optimizer):
    loss_scaler = NativeScaler()
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    return functools.partial(loss_scaler, create_graph=is_second_order)

def get_loss_scaler_v2(optimizer, parameters, clip_grad=None, clip_mode='norm', create_graph=None):
    loss_scaler = NativeScaler()
    if create_graph is None:
        create_graph = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    return functools.partial(loss_scaler,optimizer=optimizer,clip_grad=clip_grad, 
                             clip_mode = clip_mode,parameters=parameters,create_graph=create_graph)

class LossScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, optimizer, clip_grad=None, clip_mode='norm', 
                 parameters=None, create_graph=None):
        self._scaler = torch.cuda.amp.GradScaler()
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.clip_mode = clip_mode
        self.parameters = parameters
        if create_graph is None:
            create_graph = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        self.create_graph = create_graph
        assert parameters is not None

    def __call__(self, loss):
        self._scaler.scale(loss).backward(create_graph=self.create_graph)
        if self.clip_grad is not None:
            self._scaler.unscale_(self.optimizer) 
            dispatch_clip_grad(self.parameters, self.clip_grad, mode=self.clip_mode)
        self._scaler.step(self.optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_fast_tokenizer(model_name, path=''):
    if model_name == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(path,'roberta-base'), do_lower_case=True)
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(os.path.join(path,'xlnet-base-cased')) 
    else:
        tokenizer = BertWordPieceTokenizer(
            "data/.bert-base-uncased-vocab.txt",
            lowercase=True)
    return tokenizer

def get_tokenizer(model_name, path=''):
    if model_name == 'roberta':
        print('load roberta-base tokenizer')
        tokenizer = RobertaTokenizer.from_pretrained(os.path.join(path,'roberta-base'), do_lower_case=True)
    elif model_name == 'xlnet':
        print('load xlnet-base-cased tokenizer')
        tokenizer = XLNetTokenizer.from_pretrained(os.path.join(path,'xlnet-base-cased'))
    else:
        print('load bert-base-uncased tokenizer')
        tokenizer = BertTokenizer.from_pretrained(os.path.join(path,'bert-base-uncased'), do_lower_case=True)
    return tokenizer

def check_gorup(n_labels,num_group):
    num_ele_per_group = math.ceil(n_labels / num_group)
    if (num_group * num_ele_per_group - n_labels) >= num_ele_per_group:
        return check_gorup(n_labels,num_group-1)
    else:
        return [num_group, num_ele_per_group]

def get_groups(n_labels, num_group, num_ele_per_group):
    group_y , idx = [] , 0
    for i in range(num_group-1):
        group_y.append(np.array([idx + _ for _ in range(num_ele_per_group)]))
        idx += num_ele_per_group
    has_padding = False
    if n_labels - idx>0:
        last_y = np.array([idx + _ for _ in range(n_labels -idx)])
        if len(last_y)< num_ele_per_group:
            num_pad = num_ele_per_group - len(last_y)
            last_y = np.pad(last_y,(0, num_pad), 'constant', constant_values=n_labels)
            has_padding = True
            print('*** pad last group with constant mode, \
                  length = ',num_pad)
        group_y.append(last_y)
    assert len(group_y) == num_group and len(group_y[-1])==num_ele_per_group
    return torch.tensor(group_y), has_padding
    
def get_topk_candidates(group_y, group_outputs, group_labels=None, topk=10):
    outputs = torch.sigmoid(group_outputs.detach())
    if group_labels is not None: outputs += group_labels
    scores, indices = torch.topk(outputs, k=topk)
    topk, topk_scores = [], []
    for index, score in zip(indices, scores):
        topk.append(group_y[index])
        topk_scores.append([torch.full(c.shape, s) for c, s in zip(topk[-1], score)])
        topk[-1] = topk[-1].flatten()
        topk_scores[-1] = torch.cat(topk_scores[-1])
    return torch.stack(topk).long(), torch.stack(topk_scores)


def get_candidate_labels(targets,candidates):
    bs, labels = 0 , []
    _targets = targets[targets>-1]
    _targets_num  = (targets>-1).sum(-1)
    for i, n in enumerate(_targets_num):
        _label = _get_row_labels(_targets[bs: bs+n],candidates[i])
        labels.append(_label)
        bs+=n
    return torch.stack(labels)

def _get_row_labels(row_tar,row_can):
    compare = row_tar.view(1,-1) - row_can.view(-1,1) == 0
    _label = compare.sum(-1).bool().float()
    if len(row_tar) != _label.sum().item():
        k, no_tar = 0, row_tar[~compare.sum(0).bool()]
        for tar in no_tar:
            for j in range(k,_label.size(0)):
                if _label[j].item() != 1:
                    row_can[j] = tar
                    _label[j] = 1
                    k=j
                    break
    return _label


def get_element_labels_v2(targets,candidates):
    target_candidates = targets[targets>-1].detach().cpu()
    target_candidates_num  = (targets>-1).sum(-1).detach().cpu()
    candidates = candidates.detach().cpu()
    new_labels,bs = [],0
    for i, n in enumerate(target_candidates_num.numpy()):
        be = bs + n
        c = set(target_candidates[bs: be].numpy())
        c2 = candidates[i].numpy()
        new_labels.append(torch.tensor([1.0 if i in c else 0.0 for i in c2 ]))
        if len(c) != new_labels[-1].sum():
            s_c2 = set(c2)
            for cc in list(c):
                if cc in s_c2:
                    continue
                for j in range(new_labels[-1].shape[0]):
                    if new_labels[-1][j].item() != 1:
                        c2[j] = cc
                        new_labels[-1][j] = 1.0
                        break
        bs = be
    labels = torch.stack(new_labels).cuda()
    return labels, candidates.cuda()


def calculate_accuracy(logits, labels, candidates=None):
    acc1, acc3, acc5, total = 0, 0, 0, 0
    _scores, indices = torch.topk(logits.detach().cpu(), k=10)
    if candidates is not None:
        candidates = candidates.detach().cpu()
    for i, lbl in enumerate(labels.numpy()):
        if candidates is not None:
            _lbl = set(lbl[lbl>-1])
            pred = candidates[i][indices[i]].numpy()
        else:
            _lbl = set(np.nonzero(lbl)[0])
            pred = indices[i, :5].numpy()

        acc1 += len(set([pred[0]]) & _lbl)
        acc3 += len(set(pred[:3]) & _lbl)
        acc5 += len(set(pred[:5]) & _lbl)
        total += 1

    return total, acc1, acc3, acc5

class AverageMeter():
    """Computes and stores the accuracy and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.acc1 = 0
        self.acc3 = 0
        self.acc5 = 0

    def get_top_p5(self, count, acc1, acc3, acc5):
        self.count += count
        self.acc1 += acc1
        self.acc3 += acc3
        self.acc5 += acc5
        p1 = self.acc1 * 100 / self.count
        p3 = self.acc3 * 100 / (self.count * 3)
        p5 = self.acc5 * 100 / (self.count * 5)
        return p1, p3, p5

    def get_accuracy(self, logits, labels, candidates=None):
        total, acc1, acc3, acc5 = calculate_accuracy(logits,labels,candidates)
        return self.get_top_p5(total, acc1, acc3, acc5)



######################### for distribution training #################################
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
    
