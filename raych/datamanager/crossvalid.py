from collections import defaultdict
import numpy as np
import pandas as pd
import re
from sklearn import model_selection


def build_cv_stratified(dataframe, label_colunm_name, num_splits, save_path=None):
    dataframe["kfold"] = -1
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    num_bins = int(np.floor(1 + np.log2(len(dataframe))))
    dataframe[:, "bins"] = pd.cut(dataframe[label_colunm_name],
                                  bins=num_bins,
                                  labels=False)
    kf = model_selection.StratifiedKFold(n_splits=num_splits)

    for (fold, _, index) in enumerate(kf.split(X=dataframe, y=dataframe.bins.values)):
        dataframe.loc[index, "kfold"] = fold

    if save_path:
        dataframe.to_csv(save_path, index=False, encoding="utf-8")

    return dataframe



def build_data_cv_rand(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    适用于，正例和负例分别存放于两个文件中，每一行为一个sample
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            # ！！！中文注意分词，或者直接使用 char 字符，或者删掉 num_words
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0, cv)}
            revs.append(datum)

    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),  # ！！！中文注意分词，或者使用 char
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab


def clean_str(string, TREC=False):
    """
    英文处理
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()