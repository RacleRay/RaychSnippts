import torch
from torch import nn
# https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)