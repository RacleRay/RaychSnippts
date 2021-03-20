# CoordConv
# https://arxiv.org/pdf/1807.03247.pdf

"""
在Solo语义分割算法和Yolov5中使用。

论文实验发现卷积网络在坐标变换上的能力，无法将空间表示转换成笛卡尔空间中的坐标。

比如，向一个网络中输入(i, j)坐标，要求它输出一个64×64的图像，并在坐标处画一个正方形或者一个像素，然而网络在测试集上却无法完成。

分析原因是卷积作为一种局部的、共享权重的过滤器应用到输入上时，它是不知道每个过滤器在哪，无法捕捉位置信息的。
因此我们可以帮助卷积，让它知道过滤器的位置。仅仅需要在输入上添加两个通道，一个是i坐标，另一个是j坐标。
"""


import torch


def coord(x):
    in_feats = x # 当前实例特征tensor

    # 生成从-1到1的线性值
    x_range = torch.linspace(-1, 1, in_feats.shape[-1], device=in_feats.device)
    y_range = torch.linspace(-1, 1, in_feats.shape[-2], device=in_feats.device)
    y, x = torch.meshgrid(y_range, x_range) # 生成二维坐标网格

    y = y.expand([in_feats.shape[0], 1, -1, -1]) # 扩充到和in_feats相同维度
    x = x.expand([in_feats.shape[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1) # 位置特征

    in_feats = torch.cat([in_feats, coord_feat], 1) # concatnate一起作为下一个卷积的输入

    return  in_feats