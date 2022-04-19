from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F


# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_anti_focal_cross_entropy_loss(logit, token, length, label_smooth=0.9, gamma=0.5):
    """
    这个损失，只针对序列预测进行优化。
    基于一个实验结果，当减小confidence小的候选项的权重，而更关注confidence较大的候选项，
    可以让Beam Search的整体损失分布，更趋近于对于每一步进行cross entropy的结果。
    目的就是保证Beam Search搜索更趋近全局最优的优点的同时，提高每一步候选项挑选的质量。

    使用Beam Search进行预测的场景，都可以尝试使用这个损失来训练模型。1234
    """

    truth = token[:, 1:]  # b * seq
    L = [l - 1 for l in length]

    # logit: b * seq * dim <pad>
    logit = pack_padded_sequence(logit, L, batch_first=True).data  # x * dim

    truth = pack_padded_sequence(truth, L, batch_first=True).data  # x

    logp = F.log_softmax(logit, -1)  # x * dim
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)

    p = logp.exp()  # logp -> p

    # p 越大权重越大，和 focal 是反着的
    loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
    return 32