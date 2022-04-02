import torch
from torch import nn
from torch.nn import init


class PSA(nn.Module):
    "https://arxiv.org/pdf/2105.14447.pdf"
    def __init__(self, channel=512, reduction=4, num_se=4):
        super().__init__()
        self.num_se = num_se

        self.convs = []
        for i in range(num_se):
            self.convs.append(nn.Conv2d(channel // num_se,
                                        channel // num_se,
                                        kernel_size=2 * (i + 1) + 1,
                                        padding=i + 1))

        self.se_blocks = []
        for i in range(num_se):
            self.se_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channel // num_se,
                              channel // (num_se * reduction),
                              kernel_size=1,
                              bias=False), nn.ReLU(inplace=True),
                    nn.Conv2d(channel // (num_se * reduction),
                              channel // num_se,
                              kernel_size=1,
                              bias=False), nn.Sigmoid()))

        self.softmax = nn.Softmax(dim=1)

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

    def forward(self, x):
        b, c, h, w = x.size()

        #Step1:SPC module
        SPC_out = x.view(b, self.num_se, c // self.num_se, h, w)  #bs,g,ch,h,w
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        #Step2:SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        #Step3:Softmax
        softmax_out = self.softmax(SE_out)

        #Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out