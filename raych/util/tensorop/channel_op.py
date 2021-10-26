import torch


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # shuffle in groups
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def channel_shift(x, shift):
    x = torch.cat([x[:, shift:, ...], x[:, :shift, ...]], dim=1)
    return x