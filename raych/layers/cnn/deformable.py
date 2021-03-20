# Deformable Convolutional
# https://arxiv.org/pdf/1703.06211.pdf
# https://arxiv.org/pdf/1811.11168.pdf
# https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/master/modules/deform_conv.py


"""
DCN：Deformable Convolutional Net

变形卷积可以看作变形+卷积两个部分

在各大主流检测网络中，变形卷积真是涨点神器

它的“局部感受野”是可学习的，面向全图的。不需要额外的监督。

变形卷积和STN过程非常类似，STN是利用网络学习出空间变换的6个参数，对特征图进行整体变换，旨在增加网络对形变的提取能力。
DCN是利用网络学习数整图offset，比STN的变形更“全面一点”。STN是仿射变换，DCN是任意变换。


for use:
    https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch
"""


"""
psudo code:
    # 学习出offset，包括x和y两个方向，注意是每一个channel中的每一个像素都有一个x和y的offset
    offset = p_conv(x)

    # V2的时候还会额外学习一个权重系数，经过sigmoid拉到0和1之间
    if v2:
        m = torch.sigmoid(m_conv(x))

    # 利用offset对x进行插值，获取偏移后的x_offset
    x_offset = interpolate(x,offset)

    # V2的时候，将权重系数作用到特征图上
    if v2:
        m = m.contiguous().permute(0, 2, 3, 1)
        m = m.unsqueeze(dim=1)
        m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
        x_offset *= m

    # offset作用后，在进行标准的卷积过程
    out = conv(x_offset)
"""