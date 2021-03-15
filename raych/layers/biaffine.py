import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BiAffine(nn.Module):
    def __init__(self, tensorb_dim, tensora_dim):
        super(BiAffine, self).__init__()
        self.tensorb_dim = tensorb_dim
        self.tensora_dim = tensora_dim

        self.tensora_weight = Parameter(torch.Tensor(self.tensora_dim))
        self.tensorb_weight = Parameter(torch.Tensor(self.tensorb_dim))

        self.b = Parameter(torch.Tensor(1))

        self.U = Parameter(torch.Tensor(self.tensora_dim, self.tensorb_dim))

        self.init_parameters()

    def init_parameters(self):
        bound = 1 / math.sqrt(self.tensora_dim)
        nn.init.uniform_(self.tensora_weight, -bound, bound)
        bound = 1 / math.sqrt(self.tensorb_dim)
        nn.init.uniform_(self.tensorb_weight, -bound, bound)

        nn.init.constant_(self.b, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, tensora, tensorb, mask_tensora=None, mask_tensorb=None):
        """
        Y = A @ W1 @ B + A @ W2 + W3 @ B

        Args:
            tensora: shape = [batch, length_tensora, tensora_dim]
            tensorb: shape = [batch, length_tensorb, tensorb_dim]
            mask_tensora (optional): shape = [batch, length_tensora]. Defaults to None.
            mask_tensorb (optional): shape = [batch, length_tensorb]. Defaults to None.
        """
        cross = torch.matmul(tensora, self.U)
        cross = torch.matmul(cross, tensorb)

        partial_A = torch.matmul(tensora, self.tensora_weight).unsqueeze(2)
        partial_B = torch.matmul(tensorb, self.tensorb_weight).unsqueeze(1)

        out = cross + partial_A + partial_B  # [batch, length_tensora, length_tensorb]

        if mask_tensora is not None:
            out = out * mask_tensora.unsqueeze(2)
        if mask_tensorb is not None:
            out = out * mask_tensorb.unsqueeze(1)

        return out
