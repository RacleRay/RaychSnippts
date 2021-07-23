import numpy as np
import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):
    """用了两个串联的MLP结构作为memory units，直接Attention"""
    def __init__(self, d_model, d_memo=64):
        super(ExternalAttention, self).__init__()

        self.mk = nn.Linear(d_model, d_memo, bias=False)
        self.mv = nn.Linear(d_memo, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            else if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            else if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, query):
        attn = self.mk(query)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        attn = self.mv(attn)
        return attn