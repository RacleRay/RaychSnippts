import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

"""
https://openreview.net/forum?id=TVHS5Y4dNvM

作者是为了验证 视觉Transformer的强大性能是否可能更多地来自于这种基于patch的表示，还是来自于Transformer结构本身

结果，ConvMixer表现很好，论文也是实验表示，基于patch的表示可能是更重要的。

"""
def ConvMixer(dim, depth, kernel_size=9, patch_size=7, num_classes=1000):
    return nn.Sequential(
        # input process
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        # conv and mix
        *[
            nn.Sequential(
                Residual(nn.Sequential(nn.Conv2d(dim,
                                                 dim,
                                                 kernel_size=kernel_size,
                                                 groups=dim,
                                                 padding=kernel_size // 2),  # groups=dim -> depth wise, channel conv
                                       nn.GELU(),
                                       nn.BatchNorm2d(dim))),
                         nn.Conv2d(dim, dim, kernel_size=1),  # point wise, spatial position
                         nn.GELU(),
                         nn.BatchNorm2d(dim)
            ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(dim, num_classes)
    )
