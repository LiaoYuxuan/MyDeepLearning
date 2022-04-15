# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @brief      : 通用工具函数，如图像变换处理等
"""


import torch
import random
import psutil
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 用于计算当前和累计的准确率、损失大小
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # 当前值
        self.val = 0
        # 平均值
        self.avg = 0
        # 累计值
        self.sum = 0
        # 值的数量
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 设置随机数种子
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

