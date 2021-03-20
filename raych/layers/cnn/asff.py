# ASFF: Adaptively Spatial Feature Fusion
# https://arxiv.org/abs/1911.09516v1

"""
为了更加充分的利用高层语义特征和底层细粒度特征，很多网络都会采用FPN的方式输出多层特征，
但是它们都多用concat或者element-wise这种融合方式.

论文提出Adaptively Spatial Feature Fusion，即自适应特征融合方式

Feature Resizing + Adaptive Fusion
"""


import torch
import torch.nn as nn
import torch.functional as F


class ASFF(nn.Module):
    def __init__(self, level):
        super(ASFF, self).__init__()
        self.level = level
        # 输入的三个特征层的channels, 根据实际修改
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        # 每个层级三者输出通道数需要一致
        if level == 0:
            self.stride_level_1 = conv_bn_relu(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = conv_bn_relu(self.dim[2], self.inter_dim, 3, 2)
            self.expand = conv_bn_relu(self.inter_dim, 1024, 3, 1)
        elif level == 1:
            self.compress_level_0 = conv_bn_relu(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = conv_bn_relu(self.dim[2], self.inter_dim, 3, 2)
            self.expand = conv_bn_relu(self.inter_dim, 512, 3, 1)
        elif level == 2:
            self.compress_level_0 = conv_bn_relu(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = conv_bn_relu(self.dim[1], self.inter_dim, 1, 1)
            self.expand = conv_bn_relu(self.inter_dim, 256, 3, 1)

        compress_c = 8
        self.weight_level_0 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, 1, 1, 0)

    # 尺度大小 level_0 < level_1 < level_2
    def forward(self, x_level_0, x_level_1, x_level_2):
        # Feature Resizing过程
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, 2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, 4, mode='nearest')
            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(level_1_compressed, 2, mode='nearest')
            else:
                level_1_resized =F.interpolate(x_level_1, 2, mode='nearest')
            level_2_resized =x_level_2

        # 融合权重也是来自于网络学习
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)   # alpha

        # 自适应融合
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:] +\
                            level_1_resized * levels_weight[:,1:2,:,:] +\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)
        return out


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False):
    return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )