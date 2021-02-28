import random, torch, numpy as np


def prepare_seed(rand_seed, deterministic=True):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # 固定使用默认卷积算法
        torch.backends.cudnn.benchmark = False  # 关闭动态卷积算法优化
    else:
        # 当网络不会动态变化时，且输入shape固定，优化卷积计算
        torch.backends.cudnn.benchmark = True
