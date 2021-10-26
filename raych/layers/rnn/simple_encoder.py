import torch
import torch.nn as nn


class RnnEncoder(nn.Module):
    """
    可使用awdRNN

    A ``RnnEncoder`` is a rnn layer.  As a
    :class:`Seq2SeqEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, num_tokens, output_dim)``.

    Parameters
    ----------
    input_dim : ``int``
        input dimension
    output_dim: ``int``
        output dimension, which should be divided by 2 if bidirectional == true
    rnn_name" ``str``
        name of the rnn networks
    bidirectional: ``bool``, default=``True``
        whether the rnn is bidirectional
    dropout: ``float``, default=``None``
        dropout rate
    normalizer: ``str``, default = ``None``
        name of the normalization we use
    affine_for_normalizer: bool = False
        whether affine is used in the normalization
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 rnn_name: str = "lstm",
                 bidirectional: bool = True):
        super(RnnEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_name = rnn_name
        self.bidirectional = bidirectional

        if bidirectional:
            assert output_dim % 2 == 0
            hidden_size = output_dim // 2
        else:
            hidden_size = output_dim

        if rnn_name == "lstm":
            self._rnn = torch.nn.LSTM(
                input_dim,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                bias=False
            )
        else:
            self._rnn = torch.nn.GRU(
                input_dim,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                bias=False
            )

    def forward(self, input_tensors, mask=None):
        # if mask is not None:
        #     input_tensors = input_tensors * mask.unsqueeze(-1).float()

        encoded_output, _ = self._rnn(input_tensors)

        return encoded_output