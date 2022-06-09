import torchvision.models as models
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as  np
from torch.autograd import Variable
from functools import reduce
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

class ResNet(torch.nn.Module):
    def __init__(self,num_classes, name,*args, **kwargs):
        super(ResNet, self).__init__()
        if name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif name == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.net = torch.nn.Sequential(*list(resnet.children())[:-1])
        #self.classifier = nn.ModuleList([nn.Linear(2048 , num_cls) for num_cls in num_classes])
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes[0],
                                                                num_classes[1],
                                                                2048)))
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        print('resnet mhs ...+++')

    def forward(self, x, labels=None):
        h = self.net(x)
        h = h.view(-1, h.shape[1])
        if labels is not None:
            m_labels = labal_coding(labels, self.num_classes)
            logits = self._get_group_logits(h, m_labels)
            loss = self.loss_fn(logits, m_labels[:, 1])
            return loss
        else:
            outputs = h @ self.weight.view(-1, self.weight.size(-1)).T
            return outputs

    def _get_group_logits(self, logits, m_labels, num_group=10):
        outputs = []
        for i in range(logits.size(0)):
            it = m_labels[i, 0].item()
            index = np.array([_ for _ in range(0, it)] + [_ for _ in range(it + 1, self.num_classes[0])])
            np.random.shuffle(index)
            index[0] = it
            weight = self.weight[index[:num_group]].view(-1, self.weight.size(-1))
            out = weight @ logits[i]
            outputs.append(out)
        return torch.vstack(outputs).cuda()

if __name__=='__main__':
    x = torch.rand((2,3,96,96))
    net = ResNet([300,350],name='resnet18')
    outputs = net(x)
    print([out.shape for out in outputs])
