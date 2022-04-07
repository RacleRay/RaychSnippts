"""
https://arxiv.org/abs/2106.14448
https://github.com/dropreg/R-Drop/tree/main/fairseq_src/examples

https://github.com/dropreg/R-Drop/blob/3d97565595747f3b3d9c4701cb2fb824a9139913/vit_src/models/modeling.py#L290
https://github.com/dropreg/R-Drop/blob/3d97565595747f3b3d9c4701cb2fb824a9139913/huggingface_transformer_src/src/transformers/models/roberta/modeling_roberta.py#L1239
https://github.com/dropreg/R-Drop/blob/3d97565595747f3b3d9c4701cb2fb824a9139913/huggingface_transformer_src/src/transformers/models/bert/modeling_bert.py#L1545


R-Drop 作用于模型的输出层，弥补了 Dropout 在训练和测试时的不一致性。

简单来说就是在每个 mini-batch 中，每个数据样本过两次带有 Dropout 的同一个模型，R-Drop 再使用 KL-divergence 约束两次的输出一致。
所以，R-Drop 约束了由于 Dropout 带来的两个随机子模型的输出一致性。

R- Drop 只是简单增加了一个 KL-divergence 损失函数项，并没有其他任何改动。只对（子模型）网络的输出预测进行了正则约束。

实验表明，在5个常用的包含 NLP 和 CV 的任务中（一共18个数据集），R-Drop 都取得了非常不错的结果提升，
并且在机器翻译、文本摘要等任务上取得了当前最优的结果。

在实际实现中，数据 x_i 不需要过两次模型，而只需要把 x_i 在同一个 batch 中复制一份即可。


R-Drop 则在训练过程中通过刻意对于子模型之间的输出进行约束，来约束参数空间，让不同的输出都能一致，从而降低了训练和测试的不一致性。
控制模型自由度，从而更好地提升模型的泛化性能。

"""
import torch


def rdropout_kl(loss_of_logits_list, logits_list, num_labels, alpha=0.5):
    """
    loss_of_logits_list: 模型正常的loss，注意是两个复制的输入的loss之和。
    logits_list: 两个元素，分别为两个复制的输入对应的logits结果。
    """
    p = torch.log_softmax(logits_list[0].view(-1, num_labels), dim=-1)
    p_tec = torch.softmax(logits_list[0].view(-1, num_labels), dim=-1)
    q = torch.log_softmax(logits_list[-1].view(-1, num_labels), dim=-1)
    q_tec = torch.softmax(logits_list[-1].view(-1, num_labels), dim=-1)

    kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
    reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()
    loss_of_logits_list += alpha * (kl_loss + reverse_kl_loss) / 2.

    return loss_of_logits_list