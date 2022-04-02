import torch
from torch import nn
from torch.nn import init


class AFT_FULL(nn.Module):
    """
    https://arxiv.org/pdf/2105.14103v1.pdf  An Attention Free Transformer

    AFT_FULL: position_biases 就是简单的一组可学习参数
    AFT_LOCAL: position_biases 中相对位置参数有窗口大小限制，超出为0，实现更复杂，这里没实现
    """
    def __init__(self, d_model, steps=49, simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if (simple):
            # relative position biases.
            self.position_biases = torch.zeros((steps, steps))
        else:
            self.position_biases = nn.Parameter(torch.ones((steps, steps)))
        self.d_model = d_model
        self.steps = steps
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        bs, n, dim = input.shape

        q = self.fc_q(input)  #bs,n,dim
        k = self.fc_k(input).view(1, bs, n, dim)  #1,bs,n,dim
        v = self.fc_v(input).view(1, bs, n, dim)  #1,bs,n,dim

        numerator = torch.sum(
            torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v,
            dim=2)  #n,bs,dim
        denominator = torch.sum(
            torch.exp(k + self.position_biases.view(n, 1, -1, 1)),
            dim=2)  #n,bs,dim

        out = (numerator / denominator)  #n,bs,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  #bs,n,dim

        return out
