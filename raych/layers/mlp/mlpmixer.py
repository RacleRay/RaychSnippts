import torch
from torch import nn


"""
https://arxiv.org/abs/2105.01601?utm_source=aidigest&utm_medium=email&utm_campaign=155

图片中不同位置的mix叫做token-mixing
同一位置不同通道的mix叫做channel-mixing
"""

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        "MLP非线性变换"
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self,
                 tokens_mlp_dim,
                 channels_mlp_dim,
                 tokens_hidden_dim,
                 channels_hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block = MLPBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MLPBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y = self.ln(x)
        y = y.transpose(1, 2)         #(bs,channels,tokens)
        y = self.tokens_mlp_block(y)  #(bs,channels,tokens)
        ### channels mixing
        y = y.transpose(1, 2)         #(bs,tokens,channels)
        out = x + y                   #(bs,tokens,channels)
        y = self.ln(out)              #(bs,tokens,channels)
        y = out + self.channels_mlp_block(y)  #(bs,tokens,channels)
        return y


class MLPMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, tokens_mlp_dim, channels_mlp_dim,
                 tokens_hidden_dim, channels_hidden_dim):
        """
        tokens_mlp_dim: 对于图片数据，应该设置为 H/patch_size * W/patch_size 大小。
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        # input image to mlp dims
        self.embd = nn.Conv2d(3, channels_mlp_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.mlp_blocks = []
        for _ in range(num_blocks):
            self.mlp_blocks.append(
                MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim))
        self.fc = nn.Linear(channels_mlp_dim, num_classes)

    def forward(self, x):
        y = self.embd(x)  # bs, channels, h/patch_size, w/patch_size
        bs, c, ph, pw = y.shape
        y = y.view(bs, c, -1).transpose(1, 2)  # bs, tokens, channels

        if (self.tokens_mlp_dim != y.shape[1]):
            raise ValueError('Tokens_mlp_dim is not correct.')

        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)  # bs, tokens, channels

        y = self.ln(y)  # bs, tokens, channels
        y = torch.mean(y, dim=1, keepdim=False)  # bs, channels
        probs = self.fc(y)  # bs, num_classes

        return probs