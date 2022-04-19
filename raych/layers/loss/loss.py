from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch

"""
# reference: https://github.com/PistonY/torch-toolbox

include classes:
 - SigmoidCrossEntropy
 - FocalLoss
 - FocalLossSoftmax
 - L0Loss
 - LabelSmoothingLoss
 - CircleLoss
 - RingLoss
 - KnowledgeDistillationLoss
"""

def logits_distribution(pred, target, classes):
    one_hot = F.one_hot(target, num_classes=classes).bool()
    return torch.where(one_hot, pred, -1 * pred)


def reducing(ret, reduction='mean'):
    if reduction == 'mean':
        ret = torch.mean(ret)
    elif reduction == 'sum':
        ret = torch.sum(ret)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return ret


def _batch_weight(weight, target):
    "weight : (#classes), target: (#batches). 相当于根据target的index从weight中取出权值"
    return weight.gather(dim=0, index=target)


def logits_nll_loss(input, target, weight=None, reduction='mean'):
    """logits_nll_loss
    Different from nll loss, this is for sigmoid based loss.
    The difference is this will add along C(class) dim.
    和torch自带的NLLLoss相差不大
    """
    assert input.dim() == 2, 'Input shape should be (B, C).'
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'.format(
            input.size(0), target.size(0)))

    ret = input.sum(dim=-1)
    if weight is not None:
        ret = _batch_weight(weight, target) * ret
    return reducing(ret, reduction)


@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))  # (batch, classes)

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))  # (batch, classes) filled with smoothed false label
    smooth_label.scatter_(dim=1, index=true_labels.data.unsqueeze(1), value=confidence) # fill the true label position
    return smooth_label


class SigmoidCrossEntropy(_WeightedLoss):
    def __init__(self, classes, weight=None, reduction='mean'):
        super(SigmoidCrossEntropy, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes

    def forward(self, pred, target):
        zt = logits_distribution(pred, target, self.classes)
        return logits_nll_loss(-F.logsigmoid(zt), target, self.weight, self.reduction)


class CELoss(nn.Module):
    ''' Cross Entropy Loss'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output [N, M]
            target: ground truth             [N]
        '''
        eps = 1e-12
        # standard cross entropy loss
        loss = -1. * pred.gather(1, target.unsqueeze(-1)) + \
                torch.log(torch.exp(pred+eps).sum(dim=1))

        return loss.mean()

class FocalLoss(_WeightedLoss):
    def __init__(self, classes, gamma, weight=None, reduction='mean'):
        "gamma在原文中初值为2。但使用时设置为靠近0的值，效果更好一些。"
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes
        self.gamma = gamma

    def forward(self, pred, target):
        zt = logits_distribution(pred, target, self.classes)
        ret = -(1 - torch.sigmoid(zt)).pow(self.gamma) * F.logsigmoid(zt)
        return logits_nll_loss(ret, target, self.weight, self.reduction)


class FocalLossSoftmax(nn.Module):
    """
    Focal loss(https://arxiv.org/pdf/1708.02002.pdf)
    Shape:
        - input: (N, C)
        - target: (N)
        - Output: Scalar loss
    Examples:
        >>> loss = FocalLoss(gamma=2, alpha=[1.0]*7)
        >>> input = torch.randn(3, 7, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(7)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, gamma=0, alpha=None, reduction="none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = alpha

        self.reduction = reduction

    def forward(self, input, target):
        '''
        - input: (N, C), logits
        - target: (N)
        - Output: Scalar loss

        Parameters
        ----------
        input
        target

        Returns
        -------

        '''
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N, C]
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)

        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        # class weights
        if self.alpha is not None:
            # alpha[target[i]]
            at = self.alpha.gather(0, target.squeeze(-1))
            logpt = logpt * at.to(logpt.device)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    @staticmethod
    def convert_binary_pred_to_two_dimension(x, is_logits=True):
        """
        Args:
            x: (*): (log) prob of some instance has label 1
            is_logits: if True, x represents log prob; otherwhise presents prob
        Returns:
            y: (*, 2), where y[*, 1] == log prob of some instance has label 0,
                             y[*, 0] = log prob of some instance has label 1
        """
        probs = torch.sigmoid(x) if is_logits else x
        probs = probs.unsqueeze(-1)
        probs = torch.cat([1-probs, probs], dim=-1)
        logprob = torch.log(probs+1e-4)  # 1e-4 to prevent being rounded to 0 in fp16
        return logprob

    def __str__(self):
        return f"Focal Loss gamma:{self.gamma}"

    def __repr__(self):
        return str(self)


class L0Loss(nn.Module):
    """L0loss from
    "Noise2Noise: Learning Image Restoration without Clean Data"
    <https://arxiv.org/pdf/1803.04189>`_ paper.
    """
    def __init__(self, gamma=2, eps=1e-8):
        super(L0Loss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        loss = (torch.abs(pred - target) + self.eps).pow(self.gamma)
        return torch.mean(loss)


class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer. Same as LabelSmoothingLoss.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CircleLoss(nn.Module):
    r"""CircleLoss from
    `"Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    <https://arxiv.org/pdf/2002.10857>`_ paper.
    Parameters
    ----------
    m: float.
        Margin parameter for loss.
    gamma: int.
        Scale parameter for loss.
    Outputs:
        - **loss**: scalar.
    """
    def __init__(self, m, gamma):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.dp = 1 - m
        self.dn = m

    def forward(self, x, target):
        similarity_matrix = x @ x.T  # need gard here
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        negative_matrix = label_matrix.logical_not()
        positive_matrix = label_matrix.fill_diagonal_(False)

        sp = torch.where(positive_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
        sn = torch.where(negative_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))

        ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        logit_p = -self.gamma * ap * (sp - self.dp)
        logit_n = self.gamma * an * (sn - self.dn)

        logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(logit_p))
        logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(logit_n))

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss


class RingLoss(nn.Module):
    """Computes the Ring Loss from
    `"Ring loss: Convex Feature Normalization for Face Recognition"
    Parameters
    ----------
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and ring loss.
    l2_norm: bool
        Whether use l2 norm to embedding.
    weight_initializer (None or torch.Tensor): If not None a torch.Tensor should be provided.
    Outputs:
        - **loss**: scalar.
    """
    def __init__(self, lamda, l2_norm=True, weight_initializer=None):
        super(RingLoss, self).__init__()
        self.lamda = lamda
        self.l2_norm = l2_norm
        if weight_initializer is None:
            self.R = self.parameters(torch.rand(1))
        else:
            assert torch.is_tensor(weight_initializer), 'weight_initializer should be a Tensor.'
            self.R = self.parameters(weight_initializer)

    def forward(self, embedding):
        if self.l2_norm:
            embedding = F.normalize(embedding, 2, dim=-1)
        loss = (embedding - self.R).pow(2).sum(1).mean(0) * self.lamda * 0.5
        return loss


class CenterLoss(nn.Module):
    """Computes the Center Loss from
    `"A Discriminative Feature Learning Approach for Deep Face Recognition"
    <http://ydwen.github.io/papers/WenECCV16.pdf>`_paper.
    Implementation is refer to
    'https://github.com/lyakaap/image-feature-learning-pytorch/blob/master/code/center_loss.py'
    Parameters
    ----------
    classes: int.
        Number of classes.
    embedding_dim: int
        embedding_dim.
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and center loss.
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, classes, embedding_dim, lamda):
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.centers = nn.Parameter(torch.randn(classes, embedding_dim))

    def forward(self, embedding, target):
        expanded_centers = self.centers.index_select(0, target)
        intra_distances = embedding.dist(expanded_centers)
        loss = self.lamda * 0.5 * intra_distances / target.size()[0]
        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        return self.temperature**2 * torch.mean(
            torch.sum(-F.softmax(teacher_output / self.temperature) * F.log_softmax(student_output / self.temperature), dim=1))