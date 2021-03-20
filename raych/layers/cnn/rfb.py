# Receptive Field Block
# https://arxiv.org/abs/1711.07767


"""
目标区域要尽量靠近感受野中心，这会有助于提升模型对小尺度空间位移的鲁棒性。

可以看作是inception+ASPP的结合. 和ASPP类似，不过是使用了不同大小的卷积核作为空洞卷积的前置操作。
"""

import torch
import torch.nn as nn


class RFB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_channels
        inter_channels = in_channels // 8

        # 分支0：1X1卷积+3X3卷积
        self.branch0 = nn.Sequential(conv_bn_relu(in_channels, 2*inter_channels, 1, stride),
                                     conv_bn_relu(2*inter_channels, 2*inter_channels, 3, 1, visual, visual, False))
        # 分支1：1X1卷积+3X3卷积+空洞卷积
        self.branch1 = nn.Sequential(conv_bn_relu(in_channels, inter_channels, 1, 1),
                                    conv_bn_relu(inter_channels, 2*inter_channels, (3,3), stride, (1,1)),
                                    conv_bn_relu(2*inter_channels, 2*inter_channels, 3, 1, visual+1, visual+1, False))
        # 分支2：1X1卷积+3X3卷积*3代替5X5卷积+空洞卷积
        self.branch2 = nn.Sequential(conv_bn_relu(in_channels, inter_channels, 1, 1),
                                    conv_bn_relu(inter_channels, (inter_channels//2)*3, 3, 1, 1),
                                    conv_bn_relu((inter_channels//2)*3, 2*inter_channels, 3, stride, 1),
                                    conv_bn_relu(2*inter_channels, 2*inter_channels, 3, 1, 2*visual+1, 2*visual+1, False)  )

        self.ConvLinear = conv_bn_relu(6*inter_channels, out_channels, 1, 1, False)
        self.shortcut = conv_bn_relu(in_channels, out_channels, 1, stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # 尺度融合
        out = torch.cat((x0, x1, x2), 1)
        # 1X1卷积
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
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