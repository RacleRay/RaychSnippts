import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = Variable(1.0 / (1-p) * torch.bernoulli((1-p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False)
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False, do_seq_dropout=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3: # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


class StackedBRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 rnn_type=nn.LSTM,
                 concat_layers=False,
                 bidirectional=True,
                 add_feat=0):
        super(StackedBRNN, self).__init__()
        self.bidir_coef = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.hidden_size = hidden_size
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else (
                self.bidir_coef * hidden_size +
                add_feat if i == 1 else self.bidir_coef * hidden_size)
            rnn = rnn_type(in_size,
                           hidden_size,
                           num_layers=1,
                           bidirectional=bidirectional,
                           batch_first=True)
            self.rnns.append(rnn)

    @property
    def output_size(self):
        if self.concat_layers:
            return self.num_layers * self.bidir_coef * self.hidden_size
        else:
            return self.bidir_coef * self.hidden_size

    """
       Multi-layer bi-RNN

       Arguments:
           x (Float Tensor): a Float Tensor of size (batch * wordnum * input_dim).
           x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * wordnum).
           x_additional (Byte Tensor): a Byte Tensor of mask for the additional input tensor (batch * wordnum * additional_dim).
           x_out (Float Tensor): a Float Tensor of size (batch * wordnum * output_size).
    """

    def forward(self, x, x_mask, return_list=False, x_additional=None, dropout_p=0.0, do_seq_dropout=False):
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            if i == 1 and x_additional is not None:
                rnn_input = torch.cat((rnn_input, x_additional), 2)

            if dropout_p > 0:
                rnn_input = dropout(rnn_input, p=dropout_p, training=self.training, do_seq_dropout=do_seq_dropout)

            rnn_output = self.rnns[i](rnn_input)[0]
            hiddens.append(rnn_output)

        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output
