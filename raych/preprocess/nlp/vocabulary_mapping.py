import json
import os
import sys
# sys.path.append("./")
from collections import defaultdict

from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from raych.preprocess.nlp.read_embedding import get_embedding_matrix_and_vocab


class SentenceIter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for i, fname in enumerate(os.listdir(self.dirname)):
            print(fname)
            for line in open(os.path.join(self.dirname, fname), "r", encoding="utf-8"):
                yield line.strip()


def get_vocab_freq(sentence_iter, tokenizer):
    """使用BERT tokenizer，获取某个语料库的词频统计"""
    dict_vocab2freq = defaultdict(int)
    for i, sent in tqdm(enumerate(sentence_iter)):
        if not sent: continue

        tokens = tokenizer.tokenize(sent)
        for tok in tokens:
            dict_vocab2freq[tok] += 1

    return dict_vocab2freq


def get_wordvec_freq(w2v_file, escaped_words: list):
    """训练数据在训练word2vec时，排序是按照词频排序的。也可以直接使用get_vocab_freq。
    此处要保证，wordvec是对中文按字切分。因为BERT处理中文的方式就是按字切分。
    如果切分方式变化，需要重新定义。
    简单来讲，代码泛用性不高。
    """
    freq_list, _ = get_embedding_matrix_and_vocab(w2v_file, include_special_tokens=False)
    for word in escaped_words:
        freq_list.pop(word)
    return freq_list


if __name__ == '__main__':
    #################################################################################
    # prepare bert corpus frequence
    tokenizer = BertTokenizer.from_pretrained(
        "pretrained path"
    )

    corpus_folder = "bert pretrain corpus folder"
    bert_freq_save_path = "bert freq result save path"

    sentence_iter = SentenceIter(corpus_folder)
    bert_corpus_freq_dict = get_vocab_freq(sentence_iter, tokenizer)
    json.dump(
        bert_corpus_freq_dict,
        open(bert_freq_save_path, "w", encoding="utf-8"),
        ensure_ascii=False,
    )

    del bert_corpus_freq_dict["[CLS]"]
    del bert_corpus_freq_dict["[SEP]"]
    del bert_corpus_freq_dict["[UNK]"]
    del bert_corpus_freq_dict["[PAD]"]
    del bert_corpus_freq_dict["[MASK]"]

    del bert_corpus_freq_dict["，"]
    del bert_corpus_freq_dict["。"]
    del bert_corpus_freq_dict["！"]
    del bert_corpus_freq_dict["？"]

    #################################################################################
    # BERT model vocab.txt
    token_dict = {}
    with open("model path / vocab.txt", encoding="utf-8") as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    del token_dict["[CLS]"]
    del token_dict["[SEP]"]
    del token_dict["[UNK]"]
    del token_dict["[PAD]"]
    del token_dict["[MASK]"]

    del token_dict["，"]
    del token_dict["。"]
    del token_dict["！"]
    del token_dict["？"]

    #################################################################################
    # mapping user defined bert corpus words to BERT model vocab.txt
    vocab_freq_in_corpus = [
        (word, bert_corpus_freq_dict.get(word, 0))
        for word, _ in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    vocab_freq_in_corpus = sorted(
        vocab_freq_in_corpus,
        key=lambda x: x[1],
        reverse=True
    )
    ranked_vocab_in_corpus = [w[0] for w in vocab_freq_in_corpus]

    #################################################################################
    # 获取训练数据中，根据词频排序的id序列
    w2v_file = "path to word2vec file"
    ranked_vocab_in_task = get_wordvec_freq(w2v_file, ["，", "。", "！", "？"])

    #################################################################################
    # 获取训练数据中vocab与corpus中vocab的对应关系
    mapping_dict = {}
    count_unused = 0
    for wid in ranked_vocab_in_task:
        rank = ranked_vocab_in_task.index(wid)
        if rank < len(ranked_vocab_in_corpus):
            map_word = ranked_vocab_in_corpus[rank]
        else:
            map_word = "[unused%d]" % (count_unused + 1)
            count_unused += 1
            # print(map_word, wid)
        mapping_dict[wid] = map_word

    mapping_dict["，"] = "，"
    mapping_dict["。"] = "。"
    mapping_dict["！"] = "！"
    mapping_dict["？"] = "？"

    mapping_save_path = "path to save mapping json file"
    json.dump(
        mapping_dict,
        open(mapping_save_path, "w", encoding="utf-8"),
        ensure_ascii=False,
    )
