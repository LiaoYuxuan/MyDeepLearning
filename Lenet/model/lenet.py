# -*- coding: utf-8 -*-
"""
# @file name  : lenet.py
# @brief      : lenet模型定义
"""
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # 第一步：使用_init_初始化基础模块
    def __init__(self, classes):
        # 父类继承
        super(LeNet, self).__init__()
        # conv1输入为3通道的原图片，输出为6通道，卷积核大小同理
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 卷积池化后，需要将张量拉平，所以fc1的输入为16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    # 第二步：通过forward将各个基础模块连接起来
    def forward(self, x):
        # 对每一层输出的结果使用非线性激活
        out = F.relu(self.conv1(x))
        # 对conv1的卷积结果进行池化，池化核大小为2
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        # 将多个通道的张量拉平
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # 最终输出结果
        return out

    # 对以上的各部分进行参数初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()

