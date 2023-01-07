import numpy as np


# np.set_printoptions(precision=2)

def pmi(M, positive=True):
    """
    计算Pointwise Mutual Infomation.
    M : 共现矩阵，根据语料库统计
    """
    col_totals = M.sum(axis=0)
    row_totals = M.sum(axis=1)
    total = col_totals.sum()
    expected = np.outer(row_totals, col_totals) / total
    M = M / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        M = np.log(M)
    M[np.isinf(M)] = 0.0  # log(0) = 0
    if positive:
        M[M < 0] = 0.0
    return M


def get_svd_vec(M):
    M_pmi = pmi(M)
    # U 每一行为一个词的vec，每列正交表示不同的语义维度
    U, s, Vh = np.linalg.svd(M_pmi)
    return U
