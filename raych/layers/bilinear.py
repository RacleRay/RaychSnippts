import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


class BiLinear(nn.Module):
    def __init__(self, tensorb_dim, tensora_dim, outputs_dim, bias=True):
        super(BiLinear, self).__init__()
        self.tensorb_dim = tensorb_dim
        self.tensora_dim = tensora_dim
        self.outputs_dim = outputs_dim

        self.tensora_weight = Parameter(torch.Tensor(self.outputs_dim, self.tensora_dim))
        self.tensorb_weight = Parameter(torch.Tensor(self.outputs_dim, self.tensorb_dim))

        if bias:
            self.b = Parameter(torch.Tensor(self.outputs_dim))
        else:
            self.register_parameter('bias', None)

        self.U = Parameter(torch.Tensor(self.outputs_dim,
                                        self.tensora_dim,
                                        self.tensorb_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.tensora_weight)
        nn.init.xavier_uniform_(self.tensorb_weight)
        nn.init.xavier_uniform_(self.U)

        nn.init.constant_(self.b, 0.)

    def forward(self, tensora, tensorb):
        """
        YVec = aHVec @ W1 @ bVVec +  W2 @ aHVec  + W3 @ bVVec

        Args:
            tensora: shape = [dim1, dim2, ..., tensora_dim]
            tensorb: shape = [dim1, dim2, ..., tensorb_dim]
        """
        dims_prev = tensora.size()[: -1]
        dummpy_batch = int(np.prod(dims_prev))

        tensora = tensora.reshape(dummpy_batch, self.tensora_dim)
        tensorb = tensorb.reshape(dummpy_batch, self.tensorb_dim)

        cross = F.bilinear(tensora, tensorb, self.U, self.bias)
        partial_A = F.linear(tensora, self.tensora_weight)  # x A^T
        partial_B = F.linear(tensorb, self.tensorb_weight)  # x A^T

        out = cross + partial_A + partial_B  # [dummpy_batch, outputs_dim]
        out.reshape(dims_prev + (self.outputs_dim, ))

        return out
