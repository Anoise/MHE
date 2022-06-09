'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np
from torch.autograd import Variable
from copy import deepcopy
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=[2,5]):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        #num_classes = sorted(num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.classifier = nn.Linear(512*block.expansion, sum(num_classes))

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes[0],
                                                                num_classes[1],
                                                                512*block.expansion)))
        self.loss_fn =  nn.CrossEntropyLoss()
        self.indexes = np.array([_ for _ in range(num_classes[0])])
        print('resnet mhs ...')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_feature(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, labels=None):
        featrues = self.forward_feature(x)
        if labels is not None:
            m_labels = labal_coding(labels, self.num_classes)
            logits = self._get_group_logits(featrues,m_labels)
            loss = self.loss_fn(logits,m_labels[:,1])
            return loss
        else:
            outputs = featrues @ self.weight.view(-1,self.weight.size(-1)).T
            return outputs

    def _get_group_logits(self, logits, m_labels, num_group=4):
        outputs = []
        for i in range(logits.size(0)):
            it = m_labels[i, 0].item()
            index = np.array([_ for _ in range(0,it)]+[_ for _ in range(it+1,self.num_classes[0])])
            np.random.shuffle(index)
            index[0] = it
            weight = self.weight[index[:num_group]].view(-1, self.weight.size(-1))
            out = weight @ logits[i]
            outputs.append(out)
        return torch.vstack(outputs).cuda()


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(net)
    print(y.size())

if __name__ == '__main__':
    test()
