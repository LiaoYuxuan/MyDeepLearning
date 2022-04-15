# -*- coding: utf-8 -*-
"""
# @file name  : loader.py
# @brief      : 读取数据集
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from datasets.dataset import RMBDataset


DATASETS = ['RMB']

def get_train_loaders(opt):

    if opt.dataset == 'RMB':
        # 获得训练集和验证集的路径
        split_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "RMB_split"))
        if not os.path.exists(split_dir):
            raise Exception(r"数据 {} 不存在, 请先运行split.py分割数据集".format(split_dir))
        train_dir = os.path.join(split_dir, "train")
        valid_dir = os.path.join(split_dir, "valid")

        # 构建MyDataset实例
        train_data = RMBDataset(data_dir=train_dir, partition="train")
        valid_data = RMBDataset(data_dir=valid_dir, partition="valid")

        # 构建DataLoder
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=1)

        # 分类的类别数
        n_cls = 2

    else:
        raise Exception("其他数据集，暂不支持，请自己写哦")


    return train_loader, valid_loader, n_cls

def get_test_loaders(opt):

    if opt.dataset == 'RMB':
        # 获得训练集和验证集的路径
        split_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "RMB_split"))
        if not os.path.exists(split_dir):
            raise Exception(r"数据 {} 不存在, 请先运行split.py分割数据集".format(split_dir))
        test_dir = os.path.join(split_dir, "test")

        # 构建MyDataset实例
        test_data = RMBDataset(data_dir=test_dir, partition="test")

        # 构建DataLoder
        test_loader = DataLoader(dataset=test_data, batch_size=1)

        # 分类的类别数
        n_cls = 2

    else:
        raise Exception("其他数据集，暂不支持，请自己写哦")


    return test_loader, n_cls