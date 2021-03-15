#-*- coding:utf-8 -*-
# author: Racle
# project: RaychSnippts


import torch
import torch.nn.functional as F
import math
from torch import nn



def gelu(x):
    """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2)))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2)))