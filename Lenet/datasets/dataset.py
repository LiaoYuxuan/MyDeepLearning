# -*- coding: utf-8 -*-
"""
# @file name  : datasets.py
# @brief      : 对使用到的数据集及预处理进行定义
"""
from __future__ import print_function
import numpy as np
import torch
import os
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


random.seed(1)
# rmb_label = {"1": 0, "100": 1}

# 描述：人民币面额分类任务数据集
class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None, partition='train'):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # 样本标签
        self.label_name = {"1": 0, "100": 1}
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir, self.label_name)
        # 区分训练、验证和测试
        self.partition = partition
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 图像变换预处理
        if transform is None:
            if self.partition == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
        else:
            # 自定义的预处理
            self.transform = transform

    # 取出数据集中的每一个样本，并返回图像和标签
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        # 彩色图片一般转为RGB形式，颜色范围0~255
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            # 在这里做transform，转为tensor等等
            img = self.transform(img)
        return img, label

    # 数据集的大小
    def __len__(self):
        # print(len(self.data_info))
        return len(self.data_info)

    # 把路径下的所有数据集样本以列表的形式列出，便于后续选择
    @staticmethod
    def get_img_info(data_dir, rmb_label):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info