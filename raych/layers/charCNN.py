import torch.nn as nn
from collections import OrderedDict



class CharCNN(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels=None):
        super(CharCNN, self).__init__()

        # TODO: use different activation functions
        # self.active = nn.Tanh
        # self.active = nn.ReLU
        self.active = nn.ELU

        layers = list()
        kernel_size_list = [3] * (num_layers - 1)  # custom
        for i, kernel_s in enumerate(kernel_size_list):
            #  Conv1d: input size (N, Cin, L) and output (N, Cout, Lout)
            layers.append(('conv_{}'.format(i), nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_s, padding=1)))
            layers.append(('act_{}'.format(i), self.active()))
            in_channels = hidden_channels

        layers.append(('conv_top', nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)))
        layers.append(('act_top', self.active()))
        layers.append(('pool_top', nn.AdaptiveMaxPool1d(1)))
        self.net = nn.Sequential(OrderedDict(layers))

        self.init_parameters()

    def init_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)

    def forward(self, char_embeds):
        """
        char -> word, use char embeddings generate word embeddings
        Args:
            char_embeds: [batch, sent_length, char_length, in_channels]
        """
        shapes = char_embeds.size()

        char_embeds = char_embeds.reshape(-1, shapes[2], shapes[3]).transpose(1, 2)

        out_embeds = self.net(char_embeds)
        out_embeds = out_embeds.reshape(shapes[0], shapes[1], -1)

        return out_embeds