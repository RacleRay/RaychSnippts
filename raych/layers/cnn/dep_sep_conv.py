import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Conv
    """
    def __init__(self, in_ch, out_ch, kernel_size,):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        # When `groups == in_channels` and `out_channels == K * in_channels`,
        # Conv1d is depthwise convolution.
        # Kernel_size is 1 dimensional, for this is 3 dimensions word vectors input.
        # Groups lead to that each channel is convoluted by a "kernel size * 1" sized kernel individually.
        self.depthwise_conv = nn.Conv1d(
            in_channels=int(in_ch),
            out_channels=int(in_ch),
            kernel_size=int(kernel_size),
            groups=int(in_ch),
            padding=int(kernel_size // 2),
            bias=False
        )

        # Point wise conv.
        self.pointwise_conv = nn.Conv1d(
            in_channels=int(in_ch),
            out_channels=int(out_ch),
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.op = nn.Sequential(
            self.depthwise_conv,
            nn.ReLU(inplace=False),
            self.pointwise_conv,
        )

    def forward(self, x, mask=None):
        x_conv = self.op(x)
        if self.kernel_size % 2 == 0:
            x_conv = x_conv[:, :, :-1]

        return x_conv


if __name__ == "__main__":
    m  = DepthwiseSeparableConv(3, 3, 4)
    input = torch.randn(10, 3, 20)
    o = m(input)