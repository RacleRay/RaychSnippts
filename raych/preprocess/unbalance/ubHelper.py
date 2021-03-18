from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

from raych.util import logger


"""
Method 1: over_sampling, under_sampling

Method 2: eg. Focal loss

Method 3: unbalanced cost for each type of prediction
"""


def unbalance_helper(X_train, X_test, y_train, y_test,
                     imbalance_method='under_sampling'):
    """
    Args:
        imbalance_method (str, optional): over_sampling, or under_sampling. Defaults to 'under_sampling'.

    Returns:
        processed data
    """
    # 是否使用不平衡数据处理方式，上采样， 下采样， ensemble
    if imbalance_method == 'over_sampling':
        logger.info("Use SMOTE deal with unbalance data ")
        # x new = x ordi + lambda(x ordj - x ord i)      lambda 属于 (0,1)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        X_test, y_test = SMOTE().fit_resample(X_train, y_train)
    elif imbalance_method == 'under_sampling':
        logger.info("Use ClusterCentroids deal with unbalance data ")
        X_train, y_train = ClusterCentroids(random_state=0).fit_resample(X_train, y_train)
        X_test, y_test = ClusterCentroids(random_state=0).fit_resample(X_test, y_test)

    return X_train, y_train, X_test, y_test