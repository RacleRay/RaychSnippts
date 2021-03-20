# BlurPool
# https://arxiv.org/abs/1904.11486
"""
实践发现，CNN网络真的非常敏感，只要输入图片稍微改一个像素，或者平移一个像素，
CNN的输出就会发生巨大的变化，甚至预测错误。这可是非常不具有鲁棒性的。

一般情况下我们利用数据增强获取所谓的不变性。

本文研究发现，不变性的退化根本原因就在于下采样，无论是Max Pool还是Average Pool，
抑或是stride>1的卷积操作，只要是涉及步长大于1的下采样，均会导致平移不变性的丢失。

为了保持平移不变性，可以在下采样之前进行低通滤波。
将： stride = 1的max + 下采样  ==>> 变为：max + blur + 下采样
"""

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class BlurPool(nn.Module):
    def __init__(self,
                 channels,
                 filt_size=4,
                 stride=2,
                 pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size

        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]

        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # 定义一系列的高斯核
        if (self.filt_size == 1):
            a = np.array([1.,])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)  # 归一化操作，保证特征经过blur后信息总量不变

        # 非grad操作的参数利用buffer存储
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp),
                            self.filt,
                            stride=self.stride,
                            groups=inp.shape[1])
