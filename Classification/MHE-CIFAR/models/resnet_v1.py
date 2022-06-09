'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


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
            print('*** pad last group with constant mode, length = ',num_pad)
        group_y.append(last_y)
    assert len(group_y) == num_group and len(group_y[-1])==num_ele_per_group
    return torch.tensor(group_y).cuda()#, has_padding

def get_one_hot(lables, num_cls):
    one_hot = torch.zeros((len(lables),num_cls), device='cuda')
    one_hot.scatter_(1, lables.view(-1, 1).long(), 1)
    return one_hot

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=[2,5], hidden_dim=256):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        num_labels = num_classes[0] * num_classes[1]
        #num_classes = sorted(num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.group = nn.Linear(512*block.expansion, num_classes[0])
        self.classifier = nn.Linear(512*block.expansion, hidden_dim)
        self.embed = nn.Embedding(num_labels, hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        print('resnet v1 ...')
        self.group_y = get_groups(num_labels, num_classes[0], num_classes[1])
        print(self.group_y.shape,'gorup y')
        self.loss_fn = nn.CrossEntropyLoss()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_candidates(self, g_out, g_label=None):
        outputs = torch.sigmoid(g_out.detach())
        if g_label is not None:
            outputs += get_one_hot(g_label,self.num_classes[0])
        scores, indices = torch.topk(outputs, k=1)
        topk = self.group_y[indices].squeeze(1)
        scores = scores.repeat(1,self.num_classes[1])
        return topk, scores

    def forward(self, x, label=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        g_out=self.group(features)

        if label is not None:
            g_label = label//self.num_classes[1]
            local_label = label % self.num_classes[1]
            candidates, _ = self.get_candidates(g_out, g_label)
            local_logit = self._get_embed_outputs(candidates,features)
            loss_l = self.loss_fn(local_logit, local_label)
            loss_g = self.loss_fn(g_out, g_label)
            pred = torch.argmax(local_logit.data,-1).eq(local_label.data)
            g_pred = torch.argmax(g_out.data,-1).eq(g_label.data)
            return loss_l+loss_g, pred, g_pred
        else:
            candidates, scores = self.get_candidates(g_out)
            outputs = self._get_embed_outputs(candidates, features)
            outputs = torch.sigmoid(outputs) * scores
            return outputs, g_out


    def _get_embed_outputs(self, candidates, features):
        emb = self.classifier(features)
        embed_weights = self.embed(candidates)
        print(embed_weights.shape, emb.unsqueeze(-1).shape, 'xxx')
        outputs = torch.bmm(embed_weights, emb.unsqueeze(-1)).squeeze(-1)
        return outputs

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
