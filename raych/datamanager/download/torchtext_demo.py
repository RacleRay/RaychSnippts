# -*- coding: utf-8 -*-
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch
if torch.cuda.is_available():
    device = -1
else:
    device = 0

def imdb_detail_get():
    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize=(lambda x: x))
    # TEXT = data.Field(sequential=False)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    print('>>> train.fields', train.fields)
    print('>>> len(train)', len(train))
    print('>>> vars(train[0])', vars(train[0]))
    print('>>> vars(test[0])', vars(test[0]))

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)

    # print vocab information
    print('>>> len(TEXT.vocab)', len(TEXT.vocab))
    print('>>> TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    return train, test


def imdb_simple_get():
    # Approach 2:
    train_iter, test_iter = datasets.IMDB.iters(batch_size=4, device=-1)

    # print batch information
    batch = next(iter(train_iter))
    print(batch.text)
    print(batch.label)

    return train_iter, test_iter


if __name__ == "__main__":
    train, test = imdb_detail_get()
    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=3, device=-1)

    # print batch information
    batch = next(iter(train_iter))
    print(batch.text)
    print(batch.label)


