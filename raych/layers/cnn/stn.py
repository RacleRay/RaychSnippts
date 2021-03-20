# Spatial Transformer Networks
# https://arxiv.org/pdf/1506.02025.pdf
"""
虽然CNN使用sliding-window卷积操作，在一定程度上具有平移不变性。
但很多研究发现，下采样会破坏网络的平移不变性。
所以可以认为网络的不变性能力非常弱，更不用说旋转、尺度、光照等不变性。
一般我们利用数据增强来实现网络的“不变性”。

STN模块，显式将空间变换植入到网络当中，进而提高网络的旋转、平移、尺度等不变性。可以理解为“对齐”操作。
每一个STN模块由Localisation net，Grid generator和Sampler三部分组成。
Localisation net用于学习获取空间变换的参数。
Grid generator用于坐标映射。
Sampler用于像素的采集，是利用双线性插值的方式进行。

无监督
独立模块，可以在CNN的任何位置插入。
"""

import torch
import torch.nn as nn
import torch.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, spatial_dims):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims
        self.fc1 = nn.Linear(32 * 4 * 4, 1024)  # 根据自己的网络参数具体设置
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        batch_images = x
        self._in_ch = x.size(1)  # if is BCHW format

        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 2, 3)  # Localisation

        # 利用affine_grid生成采样点, Grid
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        # 将采样点作用到原始数据上
        rois = F.grid_sample(batch_images, affine_grid_points)
        return rois, affine_grid_points
