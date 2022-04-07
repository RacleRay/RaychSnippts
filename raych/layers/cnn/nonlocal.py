# Non-local Neural Networks
# https://arxiv.org/abs/1711.07971


"""
Non-Local是一种attention机制

non-local操作感受野可以很大，可以是全局区域，而不是一个局部区域
"""

import torch
import torch.nn as nn


class NonLocal(nn.Module):
    def __init__(self, channel):
        super(NonLocal, self).__init__()
        self.inter_channel = channel // 2

        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0, False)

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, False)

    def forward(self, x):
        b, c, h, w = x.size()  # [N, C, H , W]

        x_phi = self.conv_phi(x).view(b, c, -1)  # [N, C/2, H * W]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()  # [N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()  # [N, H * W, C/2]

        mul_theta_phi = torch.matmul(x_theta, x_phi)  # [N, H * W, H * W]
        mul_theta_phi = self.softmax(mul_theta_phi)  # [N, H * W, H * W]

        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)  # [N, H * W, C/2]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)  # # [N, C/2, H, W]

        mask = self.conv_mask(mul_theta_phi_g)  # 1X1卷积扩充通道数
        out = mask + x  # 残差连接
        return out