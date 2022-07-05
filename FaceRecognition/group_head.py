import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter
from copy import deepcopy
from functools import reduce
import numpy as np

def one_hot_coding(target, num):
    one_hot = torch.zeros((len(target), num)).cuda()
    one_hot.scatter_(1, target.view(-1, 1), 1)
    return one_hot


def multi_hot_coding(targets, num_classes=[3, 5]):
    t = targets[targets>-1]
    index = []
    for i in range(1, len(num_classes)):
        sum_class = reduce(lambda x, y: x * y, num_classes[i:])
        idx = t // sum_class
        t -= sum_class * idx
        one_hot = one_hot_coding(idx, num_classes[i - 1])
        index.append(one_hot)
    index.append(one_hot_coding(t, num_classes[-1]))
    return torch.hstack(index).cuda()

def labal_coding(labels, num_classes=[3, 5]):
    t = deepcopy(labels)
    index = []
    for i in range(1, len(num_classes)):
        sum_class = reduce(lambda x, y: x * y, num_classes[i:])
        idx = t // sum_class
        t -= sum_class * idx
        index.append(idx)
    index.append(t)
    return torch.stack(index).T

class ArcFaceV3(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, config):
        super(ArcFaceV3, self).__init__()
        self.scale = config.scale
        self.cos_m = math.cos(config.margin)
        self.sin_m = math.sin(config.margin)
        self.theta = math.cos(math.pi - config.margin)
        self.sinmm = math.sin(math.pi - config.margin) * config.margin
        self.easy_margin = False
        self.loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
        self.num_classes = config.num_classes

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (sum(config.num_classes), config.embedding_size)))


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        m_labels = labal_coding(labels, self.num_classes)
        norm_embeddings = F.normalize(logits)
        norm_weight_activated = F.normalize(self.weight)
        logits = F.linear(norm_embeddings, norm_weight_activated)
        index = torch.where(labels != -1)[0]
        logit_1, logit_2 = logits[:,:self.num_classes[0]], logits[:,self.num_classes[0]:]
        loss1 = self.calculate_loss(logit_1, m_labels[:,0], index)
        loss2 = self.calculate_loss(logit_2, m_labels[:,1], index)
        return loss1 + loss2


    def calculate_loss(self, logits, labels, index):
        target_logit = logits[index, labels[index].view(-1)]
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return self.loss_fn(logits, one_hot_coding(labels,logits.size(-1)))


class ArcFaceV1(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, config):
        super(ArcFaceV1, self).__init__()
        self.scale = config.scale
        self.cos_m = math.cos(config.margin)
        self.sin_m = math.sin(config.margin)
        self.theta = math.cos(math.pi - config.margin)
        self.sinmm = math.sin(math.pi - config.margin) * config.margin
        self.easy_margin = False
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes, config.embedding_size)))


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        norm_embeddings = F.normalize(logits)
        norm_weight_activated = F.normalize(self.weight)
        logits = F.linear(norm_embeddings, norm_weight_activated)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
      
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return self.loss_fn(logits, labels)


class ArcFaceMHS(torch.nn.Module):
    def __init__(self, config):
        super(ArcFaceMHS, self).__init__()
        self.scale = config.scale
        self.cos_m = math.cos(config.margin)
        self.sin_m = math.sin(config.margin)
        self.theta = math.cos(math.pi - config.margin)
        self.sinmm = math.sin(math.pi - config.margin) * config.margin
        self.easy_margin = False
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.num_classes = config.num_classes
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes[0], 
                                                                config.num_classes[1], 
                                                                config.embedding_size)))


    def _get_group_logits(self,logits,m_labels):
        outputs = []
        norm_logits = F.normalize(logits)
        for i in range(logits.size(0)):
            out = F.normalize(self.weight[m_labels[i,0]]) @ norm_logits[i]
            outputs.append(out)
        return torch.vstack(outputs).cuda()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        m_labels = labal_coding(labels, self.num_classes)
        logits = self._get_group_logits(logits,m_labels)
        index = torch.where(labels != -1)[0]
        local_label = m_labels[:,1]
        target_logit = logits[index, local_label[index].view(-1)]
      
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, local_label[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return self.loss_fn(logits, local_label)


class ArcFaceMHS_V2(torch.nn.Module):
    def __init__(self, config):
        super(ArcFaceMHS_V2, self).__init__()
        self.scale = config.scale
        self.cos_m = math.cos(config.margin)
        self.sin_m = math.sin(config.margin)
        self.theta = math.cos(math.pi - config.margin)
        self.sinmm = math.sin(math.pi - config.margin) * config.margin
        self.easy_margin = False
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.num_classes = config.num_classes
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes[0], 
                                                                config.num_classes[1], 
                                                                config.embedding_size)))

    def _get_group_logits(self,logits,m_labels):
        outputs = []
        norm_logits = F.normalize(logits)
        for i in range(logits.size(0)):
            it = m_labels[i,0].item()
            if it == 0:
                rand_it = np.random.randint(1,self.num_classes[0])
            elif it == self.num_classes[0]-1:
                rand_it = np.random.randint(0,self.num_classes[0]-1)
            else:
                rand_its = [np.random.randint(0,it), np.random.randint(it+1,self.num_classes[0])]
                np.random.shuffle(rand_its)
                rand_it = rand_its[0]
            weight = torch.cat((self.weight[m_labels[i,0]],self.weight[rand_it]),0)
            out = F.normalize(weight) @ norm_logits[i]
            outputs.append(out)
        return torch.vstack(outputs).cuda()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        m_labels = labal_coding(labels, self.num_classes)
        logits = self._get_group_logits(logits,m_labels)
        index = torch.where(labels != -1)[0]
        local_label = m_labels[:,1]
        target_logit = logits[index, local_label[index].view(-1)]
      
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, local_label[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return self.loss_fn(logits, local_label)


class ArcFaceMHS_B1(torch.nn.Module):
    def __init__(self, config):
        super(ArcFaceMHS_B1, self).__init__()
        self.scale = config.scale
        self.cos_m = math.cos(config.margin)
        self.sin_m = math.sin(config.margin)
        self.theta = math.cos(math.pi - config.margin)
        self.sinmm = math.sin(math.pi - config.margin) * config.margin
        self.easy_margin = False
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.num_classes = config.num_classes
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes[0], 
                                                                config.num_classes[1], 
                                                                config.embedding_size)))

    def _get_group_logits(self,logits,m_labels):
        slt = torch.unique(m_labels[:,0])
        position = torch.searchsorted(slt, m_labels[:,0])
        local_label = position * self.num_classes[1] + m_labels[:,1]
        slt_weight = self.weight[slt].reshape(-1,self.weight.size(-1))
        out = F.linear(F.normalize(logits) , F.normalize(slt_weight))
        return out, local_label

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        m_labels = labal_coding(labels, self.num_classes)
        logits, local_label = self._get_group_logits(logits,m_labels)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, local_label[index].view(-1)]
      
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, local_label[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return self.loss_fn(logits, local_label)