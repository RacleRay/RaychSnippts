from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F


def seq_cross_entropy_loss(logit, token, length, pad_idx):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=pad_idx)
    return loss