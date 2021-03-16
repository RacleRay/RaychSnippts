# -*- coding:utf-8 -*-

import math
import torch
from torch import nn
from ..activation import swish, gelu, mish

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

################################################################################################
### Modules of BERT

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # ones not equal to 1 exactly
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)                          # batch_size, seq_len, 1
        s = (x - u).pow(2).mean(-1, keepdim=True)             # batch_size, seq_len, 1
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)   # batch_size, seq_len, 768
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # adaptor of layer inputs
        self.query = nn.Linear(config.hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(config.hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(config.hidden_size, self.all_head_size * self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        "拆分为 多个head"
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*sz)               # (batch, pos, head, head_hid)
        return x.permute(0, 2, 1, 3)  # (batch, head, pos, head_hid)

    def forward(self, hidden_states, attention_mask, history_states=None):
        "history_states: seq2seq 时使用"
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)  # adaptor:  hidden size -> head size * num of head
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            # seq2seq 时，key 和 value 会融合历史信息
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # batch, head, position, head_dim
        key_layer = self.transpose_for_scores(mixed_key_layer)      # batch, head, position, head_dim
        value_layer = self.transpose_for_scores(mixed_value_layer)  # batch, head, position, head_dim

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask        # (batch, head, position, position)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)      # (batch, head, position, position)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (batch, head, position, head_dim)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[: -2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)# (batch, position, 768)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # (batch, position, 768)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None):
        """
        hidden: hidden of layers,  (batch, position, 768)
        mask: to attention_scores, (batch, 1, seq_len, seq_len)
        """
        self_output = self.self(input_tensor, attention_mask, history_states=history_states)
        attention_output = self.output(self_output, input_tensor)   # (batch, position, 768)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)    # (batch, position, intermediate_size)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # (batch, position, 768)
        return hidden_states


class TransformerFFN(nn.Module):
    "若 不使用 BertIntermediate + BertOutput 的输出层，可以选择全线性变换的 TransformerFFN"
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.ffn_type = config.ffn_type
        assert self.ffn_type in (1, 2)

        if self.ffn_type in (1, 2):
            self.wx0 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (2,):
            self.wx1 = nn.Linear(config.hidden_size, config.hidden_size)

        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x0 = self.wx0(x)
        if self.ffn_type == 1:
            x1 = x
        elif self.ffn_type == 2:
            x1 = self.wx1(x)
        out = self.output(x0 * x1)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)    # (batch, position, 768)
        return out


#########################################################################################
### 组合以上所有 layer

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.ffn_type = config.ffn_type  # 可选择 1 或者 2，对应不同的TransformerFFN，默认为 None
        if self.ffn_type:
            self.ffn = TransformerFFN(config)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None):
        """
        hidden: hidden of layers,  (batch, position, 768)
        mask: to attention_scores, (batch, 1, seq_len, seq_len)
        history_states: 在使用 bert seq2seq 时使用
        """
        attention_output = self.attention(hidden_states, attention_mask, history_states=history_states)
        if self.ffn_type:
            layer_output = self.ffn(attention_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output



