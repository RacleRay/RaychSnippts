# -*- coding: utf-8 -*-
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText


def load_tt_data(dataset_name, batch_size, max_seq_len, max_size=25000, dim=100, device="cuda"):
    "非bert输入形式，torchtext提供的数据集"
    TEXT = data.Field(lower=True,
                      include_lengths=True,
                      tokenize='spacy',
                      batch_first=True,
                      fix_length=max_seq_len)
    LABEL = data.Field(sequential=False, dtype=torch.float)

    # get: {'text': "xxxx", 'label': int}
    if dataset_name == "imdb":
        train, test = datasets.IMDB.splits(TEXT, LABEL)
    elif dataset_name == "sst":
        train, val, test = datasets.SST.splits(TEXT,
                                               LABEL,
                                               fine_grained=True,
                                               train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
    elif dataset_name == "trec":
        train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
    else:
        print("does not support this datset")

    # get glove word vec
    TEXT.build_vocab(train, max_size=max_size, vectors=GloVe(name='6B', dim=dim),
                     unk_init=torch.Tensor.zero_)
    LABEL.build_vocab(train)
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # get id represent: {'text': [int, int, int,...], 'label': int}
    train_iter, test_iter = data.BucketIterator.splits((train, test),
                                                       batch_size=batch_size,
                                                       device=device,
                                                       sort_within_batch=True,
                                                       repeat=False,
                                                       shuffle=True)

    label_size = len(LABEL.vocab)
    vocab_size = len(TEXT.vocab)
    embedding_dim = TEXT.vocab.vectors.size()[1]
    embeddings = TEXT.vocab.vectors
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

    return train_iter, test_iter, label_size, vocab_size, embeddings, pad_idx


def get_dataset_torchtext(name):
    "若只为拿到 torchtext 自带数据集的数据"
    TEXT = data.Field(sequential=False)
    LABEL = data.Field(sequential=False)

    # make splits for data
    if name == "imdb":
        train, test = datasets.IMDB.splits(TEXT, LABEL)
        return train, test
    elif name == "sst":
        train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=True, train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
        return train, val, test
    elif name == "trec":
        train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
        return train, test
    else:
        print("does not support this datset")
        return None
