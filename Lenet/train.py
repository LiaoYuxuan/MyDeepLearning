# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @brief      : 训练程序
"""
# Python和Pytorch自带的包导入
import os, argparse, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

# 把自己写的一些程序文件加入编译目录，有的电脑上似乎不用这一步
Lenet_dir = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(Lenet_dir)

# 自己写的文件导入
from model.lenet import LeNet
from datasets.dataset import RMBDataset
from datasets.loader import get_train_loaders
from utils.util import set_seed, AverageMeter

# 设置训练时的参数
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--model', type=str, default='lenet', help='model to use')
    parser.add_argument('--dataset', type=str, default='RMB', help='dataset to use')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # 由于mac和windows路径形式不一样，为避免报错，这里注释掉
    # parser.add_argument('--save_folder', type=str, default=None, help='path to save model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--log_interval', type=int, default=10, help='the interval to log')
    parser.add_argument('--val_interval', type=int, default=1, help='the interval to valid')

    opt = parser.parse_args()
    return opt

def main():
    # 为避免文件缺失导致bug，这里进行判断
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_lenet = os.path.join(BASE_DIR, "model", "lenet.py")
    path_tools = os.path.join(BASE_DIR, "utils", "util.py")
    assert os.path.exists(path_lenet), "{}不存在，请将lenet.py文件放到 {}".format(path_lenet, os.path.dirname(path_lenet))
    assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

    # 设置训练参数
    opt = parse_option()
    # 设置随机种子
    set_seed()

    # Step 1/5 数据集读入
    train_loader, val_loader, n_cls = get_train_loaders(opt)

    # Step 2/5 模型建立
    model = LeNet(classes=n_cls)
    model.initialize_weights()

    # Step 3/5 选择损失函数（分类任务一般用CE损失，拟合任务一般用MSE损失）
    criterion = nn.CrossEntropyLoss()

    # Step 4/5 选择优化器和学习率下降策略
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Step 5/5 训练
    train_acc_curve = list()
    valid_acc_curve = list()
    train_loss_curve = list()
    valid_loss_curve = list()
    print('==> Training...')

    for epoch in range(1, opt.epochs + 1):
        # 这里加了很多time.sleep(0.05)，目的是使输出结果顺序不乱
        time.sleep(0.05)
        time1 = time.time()
        train_loss, train_acc = train(opt, epoch, train_loader, model, criterion, optimizer, scheduler)
        time2 = time.time()
        # 输出一个epoch内的训练情况
        train_acc_curve.append(train_acc)
        train_loss_curve.append(train_loss)
        time.sleep(0.05)
        print('epoch: {}, total time: {:.2f}, train loss: {:.4f}, train acc: {:.4f}'.format(epoch, time2 - time1, train_loss, train_acc))

        # 验证模型性能
        if epoch % opt.val_interval == 0:
            time.sleep(0.05)
            print('==> Validing...')
            time.sleep(0.05)
            val_loss, val_acc = valid(opt, epoch, val_loader, model, criterion, optimizer, scheduler)
            time.sleep(0.05)
            valid_acc_curve.append(val_acc)
            valid_loss_curve.append(val_loss)
            print('epoch: {}, val loss: {:.6f}, val acc: {:.6f}'.format(epoch, val_loss, val_acc))


        # 保存模型
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = model.state_dict()
            save_file = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', 'RMB', 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)))
            torch.save(state, save_file)

    train_x = range(len(train_acc_curve))
    plt.figure(1)
    plt.plot(train_x, train_acc_curve, label='Train_acc')
    plt.plot(train_x, valid_acc_curve, label='Valid_acc')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure(2)
    plt.plot(train_x, train_loss_curve, label='Train_loss')
    plt.plot(train_x, valid_loss_curve, label='valid_loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


# 用于训练的函数
def train(opt, epoch, train_loader, model, criterion, optimizer, scheduler):

    total = 0.
    correct = 0.
    # train_loss = 0.
    train_loss = AverageMeter()
    # 设置进度条, ncols表示进度条设置的总长度
    tbar = tqdm(train_loader, ncols=130)

    model.train()

    for i, data in enumerate(tbar):

        # 前向传播
        inputs, labels = data
        outputs = model(inputs)

        # 反向传播
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        train_loss.update(loss.item())
        # train_curve.append(loss.item())
        if i % opt.log_interval == 0:
            # tbar.set_description("Training: Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            #     epoch, opt.epochs, train_loss.avg, correct / total))
            tbar.set_description("Training: Epoch[{:0>3}/{:0>3}]".format(
                epoch, opt.epochs))
        # 更新学习率
        scheduler.step()
    return train_loss.avg, correct / total


# 用于验证的函数
def valid(opt, epoch, val_loader, model, criterion, optimizer, scheduler):
    correct_val = 0.
    total_val = 0.
    val_loss = AverageMeter()
    tbar = tqdm(val_loader, ncols=130)
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(tbar):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().numpy()

            val_loss.update(loss.item())

        # valid_curve.append(loss_val)
            tbar.set_description("Valid: Epoch[{:0>3}/{:0>3}]".format(
            epoch, opt.epochs))
    return val_loss.avg, correct_val / total_val

if __name__ == '__main__':
    main()
