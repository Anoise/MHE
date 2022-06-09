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
from models.resnet_mhp import ResNet18
from my_cifar10 import CIFAR10, CIFAR100
from torch.autograd import Variable
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument("--img-size", default=32, type=int)
    parser.add_argument("--dataset", default='c100', type=str,choices=['c100','c10'])
    parser.add_argument("--data-path", default='../../Data/cifar100', type=str,
                        help='the data path of the dataset')
    parser.add_argument("--lr", default=0.1, type=float, help='learning rate')
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum')
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--eta-min", default=1e-5, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--num-classes", nargs='*', default=[10,10], type=int, help='the number of classifier heads')
    parser.add_argument('--save-path', type=str, default='checkpoint',  help='save_path')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', default=False)
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    return args


def loss_fn(predicts, targets):
    
    one_hot_labels=[]
    for i, num_cls in enumerate(args.num_classes):
        one_hot = torch.zeros((len(predicts),num_cls), device='cuda')
        one_hot.scatter_(1, targets[:,i].view(-1, 1).long(), 1)
        one_hot_labels.append(one_hot)
    one_hot_labels = torch.hstack(one_hot_labels)
    
    # soft_y = torch.log_softmax(predicts,-1)
    # loss = -torch.sum(one_hot_labels*soft_y)/len(predicts)
    loss = F.multilabel_soft_margin_loss(predicts,one_hot_labels)
    return loss


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pred = None
        for i, out in enumerate(outputs.split(args.num_classes,-1)):
            temp = torch.argmax(out.data, 1).eq(targets[:, i].data)
            pred = pred & temp if pred != None else temp 

        correct += pred.cpu().sum().item()

    _loss = train_loss / len(trainloader)
    _acc = 100. * correct / len(trainset)
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (_loss, _acc, correct, len(trainset)))
    return [_loss, _acc]


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
        loss = loss_fn(outputs, targets)
        test_loss += loss.item()

        pred = None
        for i, out in enumerate(outputs.split(args.num_classes,-1)):
            temp = torch.argmax(out.data, -1).eq(targets[:, i].data)
            pred = pred & temp if pred != None else temp

        correct += pred.cpu().sum().item()

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
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        torch.save(state, model_path)
        best_acc = _acc
    return [_loss, _acc]


if __name__ == '__main__':

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    model_path = args.save_path +'/ckpt.t7'

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

    if args.dataset =='c100':
        trainset = CIFAR100(root=args.data_path, num_classes=args.num_classes, train=True, download=True,
                            transform=transform_train)
        testset = CIFAR100(root=args.data_path, num_classes=args.num_classes, train=False, download=True,
                           transform=transform_test)
    else:
        trainset = CIFAR10(root=args.data_path, num_classes=args.num_classes, train=True, download=True,
                            transform=transform_train)
        testset = CIFAR10(root=args.data_path, num_classes=args.num_classes, train=False, download=True,
                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.save_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        net = ResNet18(num_classes=args.num_classes)

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    end_epoch = start_epoch + args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, end_epoch, eta_min=args.eta_min)

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
    print('best acc', best_acc)


    np.save(args.save_path+'/train_infos.npy', train_infos)
    np.save(args.save_path+'/test_infos.npy', test_infos)
    print('train test infos saved ...')
