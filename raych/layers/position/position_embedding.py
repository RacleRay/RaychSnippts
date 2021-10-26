import numpy as np
import torch
from torch import nn


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class SinusoidPositionalEmbedding(torch.nn.Module):
    """
    SinusoidPositionalEmbedding: BERT 原论文position encode方法
    """
    def __init__(self, max_len=512, embed_dim=300) -> None:
        super(SinusoidPositionalEmbedding, self).__init__()

        self.encoder_position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                max_len,
                embed_dim,
                padding_idx=0
            ),
            freeze=True
        )

    def forward(self, input_pos_tensors: torch.Tensor):
        """
        input_pos_tensors : (batch_size, num_tokens).
        """
        return self.encoder_position_enc(input_pos_tensors)


class LearnedPositionalEmbedding(torch.nn.Module):
    """
    LearnedPositionalEmbedding: 作为参数学习
    """

    def __init__(self, max_len=512, embed_dim=300, positional_embedding=None) -> None:
        super(LearnedPositionalEmbedding, self).__init__()

        if positional_embedding:
            self.encoder_position_enc = positional_embedding
        else:
            self.encoder_position_enc = nn.Embedding(
                max_len,
                embed_dim,
                padding_idx=0
            )

    def forward(self, input_pos_tensors: torch.Tensor):  # pylint: disable=arguments-differ
        """
        input_pos_tensors : (batch_size, num_tokens).
        """
        return self.encoder_position_enc(input_pos_tensors)