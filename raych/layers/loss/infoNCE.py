import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class InfoNCE_Loss(nn.Module):
    """https://github.com/loeweX/Greedy_InfoMax.  Paper: Putting An End to End-to-End: Gradient-Isolated Learning of Representations

    more contrast learning mathod: https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/main_contrast.py
    """
    def __init__(self, opt, in_channels, out_channels):
        super().__init__()
        self.opt = opt
        self.negative_samples = self.opt.negative_samples
        self.k_predictions = self.opt.prediction_step

        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.k_predictions))

        self.contrast_loss = ExpNLLLoss()

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, )):
                if m in self.W_k:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="tanh"
                    # )
                    makeDeltaOrthogonal(
                        m.weight,
                        nn.init.calculate_gain("Sigmoid"),
                    )

    def forward(self, z, c, skip_step=1):
        "z可以是输入某layer前的表示，c是layer输出的表示，得到纵向的info差异。同时，z和c可以是同一layer的输出，计算不同时间步的横向info差异"
        batch_size = z.shape[0]

        total_loss = 0

        if self.opt.device.type != "cpu":
            cur_device = z.get_device()
        else:
            cur_device = self.opt.device

        # For each element in c, contrast with elements below
        for k in range(1, self.k_predictions + 1):
            ## compute log f(c_t, x_{t+k}) = z^T_{t+k} W_k c_t
            # compute z^T_{t+k} W_k:
            ztwk = (
                self.W_k[k - 1].forward(z[:, :, (k + skip_step):, :])  # B, C , H , W
                .permute(2, 3, 0, 1)  # H, W, B, C
                .contiguous())  # H, W, B, C

            ztwk_shuf = ztwk.view(ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3])  # H * W * batch, C

            # Sample more n negative_samples
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # H *  W * batch
                (ztwk_shuf.shape[0] * self.negative_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(ztwk_shuf,
                                     dim=0,
                                     index=rand_index,
                                     out=None)  # H * W * B * n, C

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.negative_samples,
                ztwk.shape[3],
            ).permute(0, 1, 2, 4, 3)  # H, W, B, C, n

            ### Compute  x_W1 . c_t:
            context = (c[:, :, :-(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2))  # H, W, B, 1, C

            # 以前 k + skip_step 步的信息，来predict后面 t 步的表示
            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(-2)  # H, W, B, 1
            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # H, W, B, n

            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # H, W, B, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # B, 1+n, H, W

            log_fk = torch.softmax(log_fk, dim=1)

            # 不同时间步 t 的差异（互信息），随机负例的互信息目标都是要最大化
            true_f = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # B, H, W

            total_loss += self.contrast_loss(input=log_fk, target=true_f)

        total_loss /= self.k_predictions

        return total_loss


class ExpNLLLoss(_WeightedLoss):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce,
                                         reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        W = torch.log(input + 1e-11)
        return F.nll_loss(W,
                          target,
                          weight=self.weight,
                          ignore_index=self.ignore_index,
                          reduction=self.reduction)


def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    """
     input = QR with Q being an orthogonal matrix or batch of orthogonal matrices and
    R being an upper triangular matrix or batch of upper triangular matrices.
    """
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")

    weights.data.fill_(0)
    dim = max(rows, cols)

    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2

    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
        weights.mul_(gain)