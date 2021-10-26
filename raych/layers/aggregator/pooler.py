import torch
from torch import nn
import torch.nn.functional as F
from raych.util.tensorop.mask import replace_masked_values, masked_softmax, weighted_sum


__all__ = ['AvgPoolerAggregator', 'DynamicRoutingAggregator', 'MaxPoolerAggregator', 'SelfAttnAggregator']


class AvgPoolerAggregator(nn.Module):
    """
    A ``AvgPoolerAggregator`` is a avg pooling layers.
    The input to this module is of shape ``(batch_size, num_tokens, input_dim)``,
    and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.
    """

    def __init__(self, ) -> None:
        super(AvgPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_max_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        if mask is not None:
            input_tensors = replace_masked_values(input_tensors, mask.unsqueeze(2), 0)

        tokens_avg_pooled = torch.mean(input_tensors, 1)

        return tokens_avg_pooled


class MaxPoolerAggregator(torch.nn.Module):
    """
    A ``MaxPoolerAggregator`` is a max pooling layers.
    the input to this module is of shape ``(batch_size, num_tokens, input_dim)``,
    and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, ) -> None:
        super(MaxPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_max_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        if mask is not None:
            input_tensors = replace_masked_values(
                input_tensors, mask.unsqueeze(2), -1e7
            )

        input_max_pooled = torch.max(input_tensors, 1)[0]

        return input_max_pooled



class SelfAttnAggregator(nn.Module):
    """
    A ``SelfAttnAggregator`` is a self attn layers.
    the input to this module is of shape ``(batch_size, num_tokens, input_dim)``,
    and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, output_dim,
                 attn_vector=None) -> None:
        super(SelfAttnAggregator, self).__init__()

        self.output_dim = output_dim

        self.attn_vector = None
        if attn_vector:
            self.attn_vector = attn_vector
        else:
            self.attn_vector_proj = nn.Linear(self.output_dim, 1)

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_self_attn_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        if self.attn_vector:
            self_attentive_logits = self.attn_vector
        else:
            self_attentive_logits = self.attn_vector_proj(input_tensors).squeeze(2)
        self_weights = masked_softmax(self_attentive_logits, mask)
        input_self_attn_pooled = weighted_sum(input_tensors, self_weights)

        return input_self_attn_pooled



def squash(input_tensors, dim=2):
    """
    Squashing function for Capsule Net`s routing algorithm.
    Parameters
    ----------
    input_tensors : a tensor
    dim: dimensions along which to apply squashing

    Returns
    -------
    squashed : torch.FloatTensor
        A tensor of shape ``(batch_size, num_tokens, input_dim)`` .
    """
    norm = torch.norm(input_tensors, 2, dim=dim, keepdim=True)   # [batch_size, out_caps_num, 1]
    norm_sq = norm ** 2   # [batch_size, out_caps_num, 1]
    s = norm_sq / (1.0+norm_sq) * input_tensors / torch.sqrt(norm_sq + 1e-8)
    return s


class DynamicRoutingAggregator(nn.Module):
    """
    A ``DynamicRoutingAggregator`` is a dynamic routing layers. the input to this module
    is of shape ``(batch_size, num_tokens, input_dim)``,
    and the output is of shape ``(batch_size, output_dim)``,
    where not necessarily input_dim == output_dim.

    Parameters
    ----------
    input_dim : ``int``
        the hidden dim of input
    out_caps_num: `` int``
        num of caps
    out_caps_dim: `` int ``
        dim for each cap
    iter_num": `` int ``
        num of iterations
    """

    def __init__(self, input_dim: int,
                 out_caps_num: int,
                 out_caps_dim: int,
                 iter_num: int = 3,
                 output_format: str = "flatten",
                 activation_function: str = "tanh",
                 device=False,
                 shared_fc=None) -> None:
        super(DynamicRoutingAggregator, self).__init__()
        self.input_dim = input_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.iter_num = iter_num
        self.output_format = output_format
        self.activation_function = activation_function
        self.device = device

        if shared_fc:
            self.shared_fc = shared_fc
        else:
            self.shared_fc = nn.Linear(input_dim, out_caps_dim * out_caps_num)

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).

        Returns
        -------
        output_tensors : torch.FloatTensor
            if "flatten":
                return tensor of shape ``(batch_size, out_caps_num * out_caps_dim)`` .
            else:
                return tensor of shape ``(batch_size, out_caps_num, out_caps_dim)``
        """
        # shared caps
        batch_size = input_tensors.size()[0]
        num_tokens = input_tensors.size()[1]

        shared_info = self.shared_fc(input_tensors)   # [batch_size, num_tokens, out_caps_dim * out_caps_num]
        if self.activation_function == "tanh":  # 效果更好，适合特征输出变换
            shared_info = torch.tanh(shared_info)
        elif self.activation_function == "relu":
            shared_info = F.relu(shared_info)

        shared_info = shared_info.view([-1, num_tokens,
                                        self.out_caps_num,
                                        self.out_caps_dim])
        # prepare mask
        # print("mask: ", mask.size())

        assert len(mask.size()) == 2
        # [bsz, seq_len, 1]
        mask_float = torch.unsqueeze(mask, dim=-1).to(torch.float32)

        B = torch.zeros(
            [batch_size, num_tokens, self.out_caps_num],
            dtype=torch.float32
        ).to(self.device)

        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, self.out_caps_num)
        B = B.masked_fill((1 - mask_tiled).byte(), -1e32)

        for i in range(self.iter_num):
            C = F.softmax(B, dim=2)
            C = C * mask_float     # (batch_size, num_tokens, out_caps_num)
            C = torch.unsqueeze(C, dim=-1)     # (batch_size, num_tokens, out_caps_num, 1)

            # 公式中 mij
            weighted_uhat = C * shared_info  # [batch_size, num_tokens, out_caps_num, out_caps_dim]

            S = torch.sum(weighted_uhat, dim=1)  # [batch_size, out_caps_num, out_caps_dim]

            V = squash(S, dim=2)     # [batch_size, out_caps_num, out_caps_dim]
            V = torch.unsqueeze(V, dim=1)    # [batch_size, 1, out_caps_num, out_caps_dim]

            # .detach()：B 只用作计算V与U的相关性，不进行BP
            B += torch.sum((shared_info * V).detach(), dim=-1)     # [batch_size, num_tokens, out_caps_num]

        V_ret = torch.squeeze(V, dim=1)  # (batch_size, out_caps_num, out_caps_dim)

        if self.output_format == "flatten":
            V_ret = V_ret.view([-1, self.out_caps_num * self.out_caps_dim])

        return V_ret



if __name__ == "__main__":
    #############################################################
    # test DynamicRoutingAggregator
    batch_size = 3
    num_tokens = 4
    out_caps_num = 2
    out_caps_dim = 10
    input_dim = 10
    mask = [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]
    mask = torch.tensor(mask)

    input_tensors_ = torch.randn([3, 4, 10])

    dr = DynamicRoutingAggregator(
        input_dim,
        out_caps_num,
        out_caps_dim,
        iter_num=3,
        output_format="flatten"
    )

    V_ret = dr(input_tensors_, mask)