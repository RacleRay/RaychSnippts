# Ghost module
# https://arxiv.org/pdf/1911.11907.pdf


"""
在ImageNet的分类任务上，GhostNet在相似计算量情况下Top-1正确率达75.7%，高于MobileNetV3的75.2%

在CNN模型中，特征图是存在大量的冗余，当然这也是非常重要和有必要的。

Ghost就是为了降低卷积的通道数，然后利用某种变换生成冗余的特征图。
一个仅通过少量计算（论文称为cheap operations）就能生成大量特征图的结构——Ghost Module。

使用比原始更少量卷积运算，比如正常用64个卷积核，这里就用32个，减少一半的计算量。
利用深度分离卷积，从上面生成的特征图中变换出冗余的特征。
"""

import math
import torch
import torch.nn as nn


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        inter_channels = math.ceil(out_channels / ratio)
        new_channels = inter_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        # cheap操作，注意利用了分组卷积进行通道分离
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_channels, new_channels, dw_size, 1, dw_size//2, groups=inter_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)       # 减少输出channel数的卷积操作
        x2 = self.cheap_operation(x1)   # cheap变换操作增加channel
        out = torch.cat([x1,x2], dim=1) # 二者cat到一起
        return out[:, :self.out_channels, :, :]