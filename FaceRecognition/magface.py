import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter
from copy import deepcopy
from functools import reduce

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
        #self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.num_classes = config.num_classes

        #self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes, config.embedding_size)))
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


class ArcFaceGroup(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, config):
        super(ArcFaceGroup, self).__init__()
        self.scale = config.scale
        self.cos_m = math.cos(config.margin)
        self.sin_m = math.sin(config.margin)
        self.theta = math.cos(math.pi - config.margin)
        self.sinmm = math.sin(math.pi - config.margin) * config.margin
        self.easy_margin = False
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes[-1], config.num_classes[0], config.embedding_size)))


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

class ArcFaceV2(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, config):
        super(ArcFaceV2, self).__init__()
        # feat_dim, num_class, margin_arc=0.35, margin_am=0.0, scale=32
        self.weight = Parameter(torch.Tensor(config.embedding_size, config.num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = config.margin_arc
        self.margin_am = config.margin_am
        self.scale = config.scale
        self.cos_margin = math.cos(config.margin_arc)
        self.sin_margin = math.sin(config.margin_arc)
        self.min_cos_theta = math.cos(math.pi - config.margin_arc)
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return self.loss_fn(output, labels)





class CircleLoss(nn.Module):
    def __init__(self, config) -> None:
        super(CircleLoss, self).__init__()
        self.m = config.margin
        self.gamma = config.gamma
        self.soft_plus = nn.Softplus()
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (config.num_classes, config.embedding_size)))

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:

        norm_embeddings = F.normalize(logits)
        norm_weight_activated = F.normalize(self.weight)
        logits = F.linear(norm_embeddings, norm_weight_activated)

        #logits = F.normalize(logits)

        sp, sn  = self.convert_label_to_similarity(logits, labels)

        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


    def convert_label_to_similarity(self, normed_feature: torch.Tensor, label: torch.Tensor):
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class MagFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, config):
        super(MagFace, self).__init__()
        self.weight = Parameter(torch.Tensor(config.embedding_size, config.num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_am = config.margin_am
        self.scale = config.scale        
        self.l_a = config.l_a
        self.u_a = config.u_a
        self.l_margin = config.l_margin
        self.u_margin = config.u_margin
        self.lamda = config.lamda
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
    

    def calc_margin(self, x):
        margin = (self.u_margin-self.l_margin) / \
                 (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def forward(self, feats, labels):
        x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(self.l_a, self.u_a)# l2 norm
        ada_margin = self.calc_margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        loss_g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin)        
        cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        loss = self.loss_fn(output, labels) + torch.mean(self.lamda*loss_g)
        return loss
