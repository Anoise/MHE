#!/usr/bin/env python
"""
@author : Daojun Liang
@email  : daojunliang@gmail.com
@time   : 2020/9/21 23:40
@desc   : __init__.py.py
"""
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.resnet import ResNet18
from torchvision.datasets import CIFAR10, CIFAR100
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime

def log(s,path=None):
    '''use log to replace the print to record the useful information'''
    print(s)
    path='train.log' if path is None else path
    with open(path, 'a') as f:
        f.write(str(datetime.now()) + ': ' + s + '\n')
        f.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', default=False)
    # parser.set_defaults(resume=False)

    parser.add_argument("--img-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()

    return args


def loss_fn(predicts, targets, num_classes):
    
    one_hot_labels=[]
    soft_codings=[]
    for i, num_cls in enumerate(num_classes):
        one_hot = torch.zeros((len(targets),num_cls), device='cuda')
        one_hot.scatter_(1, targets[:,i].view(-1, 1).long(), 1)
        one_hot_labels.append(one_hot)
        soft_codings.append(torch.softmax(predicts[i],-1))
    m_hot_labels = torch.hstack(one_hot_labels)
    m_soft_code = torch.hstack(soft_codings)

    # loss = F.binary_cross_entropy(m_soft_code,m_hot_labels)
    loss = F.binary_cross_entropy_with_logits(m_soft_code,m_hot_labels)
    return loss


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        loss = net(inputs,targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    _loss = train_loss / len(trainloader)
    print('Loss: %.3f ' % (_loss))
    return [_loss]


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    _loss = test_loss / len(testloader)
    _acc = 100. * correct / len(testset)
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
          (_loss, _acc, correct, len(testset)))

    # Save checkpoint.

    if _acc > best_acc:
        print('Saving..')
        state = {
            'net': net if use_cuda else net,
            'acc': _acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, model_path)
        best_acc = _acc
    return [_loss, _acc]


### defined for adjust learning rate


if __name__ == '__main__':

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    model_path = './checkpoint/ckpt.t7'
    num_classes = [10,10]

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR100(root='../../Data/cifar100', train=True, download=True,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = CIFAR100(root='../../Data/cifar100',  train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        # net = ResNet(in_planes=48)
        net = ResNet18(num_classes=num_classes)

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # train 200 epoch
    end_epoch = start_epoch + 400
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(end_epoch), eta_min=0.00001)

    train_infos = []
    test_infos = []
    for epoch in range(start_epoch, end_epoch):
        train_info = train(epoch)
        test_info = test(epoch)
        train_infos.append(train_info)
        test_infos.append(test_info)
        # adjust_learning_rate(optimizer, epoch, end_epoch)
        scheduler.step()
        lr = scheduler.get_last_lr()
        print('lr ', lr)
    log(str('best acc={:.4f}'.format(best_acc)))

    np.save('checkpoint/train_infos.npy', train_infos)
    np.save('checkpoint/test_infos.npy', test_infos)
    print('train test infos saved ...')
