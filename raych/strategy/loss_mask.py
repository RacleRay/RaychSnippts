import torch


def get_tsa_thresh(schedule, global_step, num_train_steps, K, device):
    """Training Signal Annealing (TSA)，在训练时逐步释放训练信号. """
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))

    if schedule == 'linear':
        threshold = training_progress
    elif schedule == 'exp':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)

    output = threshold * (1 - 1 / K) + 1 / K

    return output.to(device)


def get_loss_mask(logits, loss, labels, global_steps, total_steps, device):
    """不同于UDA原论文中使用预测概率来判断根据 threshhold 的取舍，这里使用 loss 值进行取舍。"""

    K = logits.shape[-1]
    tsa_thresh = get_tsa_thresh('log', global_steps, total_steps, K, device)

    # 只保留小于阈值的 loss
    threshold_filte = torch.exp(-loss) > tsa_thresh
    loss_mask = torch.ones_like(labels, dtype=torch.float32) * (1 - threshold_filte.type(torch.float32))

    return loss_mask