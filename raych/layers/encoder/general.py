import torch.nn as nn
from raych.layers.cnn.dep_sep_conv import DepthwiseSeparableConv
from raych.layers.rnn.simple_encoder import RnnEncoder


class TextCnnEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_channel, dropout_rate=0.0, norm=False):
        super(TextCnnEncoder, self).__init__()
        self.norm = norm

        # 四个卷积核
        self.ops = nn.ModuleList()
        for kernel_size in [1, 3, 5, 7]:
            op_ = DepthwiseSeparableConv(
                embed_dim,
                hidden_channel,
                kernel_size,
            )
            self.ops.append(op_)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.LayerNorm = nn.LayerNorm(hidden_channel)

    def forward(self, input_tensors=None,
                attention_mask=None,
                position_ids=None,
                **kwargs):
        tmp_outputs = []
        for i, op in enumerate(self.ops):
            input_tensors_conv = op(input_tensors)
            tmp_outputs.append(input_tensors_conv)

        output_tensors = sum(tmp_outputs)
        output_tensors = self.dropout(output_tensors)
        if self.norm:
            output_tensors = self.LayerNorm(output_tensors)
            # output_tensors = self.LayerNorm(output_tensors + input_tensors)

        return output_tensors


class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0., norm=False):
        super(BiLSTMEncoder, self).__init__()
        self.norm = norm

        self.op = RnnEncoder(
            embed_dim,
            hidden_dim,
            rnn_name="lstm",
            bidirectional=True
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, input_tensors=None,
                attention_mask=None,
                position_ids=None,
                **kwargs):

        output_tensors = self.op(input_tensors)
        output_tensors = self.dropout(output_tensors)
        if self.norm:
            # output_tensors = self.LayerNorm(output_tensors + input_tensors)
            output_tensors = self.LayerNorm(output_tensors)

        return output_tensors
