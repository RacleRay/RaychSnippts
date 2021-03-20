# CBAM: Convolutional Block Attention Module
# https://arxiv.org/abs/1807.06521


"""
该种attention方法法只关注了通道层面上哪些层会具有更强的反馈能力.

将attention同时运用在channel和spatial两个维度上.


"""

import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=2):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, 3, 1)

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        # channel attention
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        out = x * out.expand_as(out)

        # spatial attention
        avg_out2 = torch.mean(out, dim=1, keepdim=True)
        max_out2 = torch.max(out, dim=1, keepdim=True)
        out2 = torch.cat([avg_out2, max_out2], dim=1)
        out2 = self.conv(out2)
        out2 = self.sigmoid(out2)

        out = out * out.expand_as(out2)

        return out