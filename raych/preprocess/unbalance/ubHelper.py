from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from raych.util import logger


"""
Method 1: over_sampling, under_sampling

Method 2: eg. Focal loss

Method 3: unbalanced cost for each type of prediction
"""

"""
Package Doc Link: https://imbalanced-learn.org/dev/references/generated/imblearn.combine.SMOTETomek.html

SMOTE: Use neighbors to interpolate new samples.
Tomek: Use Tomek Links to remove the pairs of two classes samples that are around the classification boundary.
ClusterCentroids: Use N cluster centroids to replace the origin samples.

NearMiss:
    (Negative sample refers to the samples from the minority class.)
    NearMiss-1 selects the positive samples for which the average distance to the N closest samples of the negative class is the smallest.
    NearMiss-2 selects the positive samples for which the average distance to the N farthest samples of the negative class is the smallest.
    NearMiss-3：对于每个少数类样本选择K个最近的多数类样本，目的是保证每个少数类样本都被多数类样本包围.

More choices:
    over_sampling: https://imbalanced-learn.org/dev/over_sampling.html#
    under_sampling: https://imbalanced-learn.org/dev/under_sampling.html
    Classifier including inner balancing samplers: https://imbalanced-learn.org/dev/ensemble.html
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
        print("Use SMOTETomek deal with unbalance data ")
        # 插值生成新样本
        X_train, y_train = SMOTETomek().fit_resample(X_train, y_train)
        X_test, y_test = SMOTETomek().fit_resample(X_train, y_train)
    elif imbalance_method == 'under_sampling':
        print("Use ClusterCentroids deal with unbalance data ")
        X_train, y_train = ClusterCentroids(random_state=0).fit_resample(X_train, y_train)
        X_test, y_test = ClusterCentroids(random_state=0).fit_resample(X_test, y_test)

    return X_train, y_train, X_test, y_test