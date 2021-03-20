# Squeeze-and-Excitation Networks
# https://arxiv.org/pdf/1709.01507.pdf


"""
一种通道注意力机制。由于特征压缩和FC的存在，其捕获的通道注意力特征是具有全局信息的。


"""

import torch
import torch.nn as nn

class SE_Block(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze
        y = self.fc(y).view(b, c, 1, 1) # Excitation: FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)       # Reweight