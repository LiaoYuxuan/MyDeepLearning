# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @brief      : 训练程序
"""
# Python和Pytorch自带的包导入
import os, argparse, sys, time
import numpy as np
import torch

# 把自己写的一些程序文件加入编译目录，有的电脑上似乎不用这一步
Lenet_dir = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(Lenet_dir)

# 自己写的文件导入
from model.lenet import LeNet
from datasets.dataset import RMBDataset
from datasets.loader import get_test_loaders
from utils.util import set_seed, AverageMeter

def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--model', type=str, default='lenet', help='model to use')
    parser.add_argument('--model_path', type=str, default='results/RMB/ckpt_epoch_10.pth', help='model path to use')
    parser.add_argument('--dataset', type=str, default='RMB', help='dataset to use')

    opt = parser.parse_args()
    return opt

def main():

    opt = parse_option()
    # 将训练好的模型权重读入
    ckpt = torch.load(opt.model_path)
    test_loader, n_cls = get_test_loaders(opt)

    if opt.model == 'lenet':
        model = LeNet(classes=n_cls)
        model.load_state_dict(ckpt)
    else:
        raise Exception("暂不支持其他模型")

    print('==> Testing...')
    correct_test = 0.
    total_test = 0.
    for i, data in enumerate(test_loader):
        # forward
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted == labels).squeeze().sum().numpy()

    print(total_test)
    print("test acc: {:.4f}".format(correct_test / total_test))

if __name__ == '__main__':
    main()