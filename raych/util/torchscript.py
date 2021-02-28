def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)