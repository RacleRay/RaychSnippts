#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eda.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''


import re
import jieba
import random
from random import shuffle
import synonyms as sy
from pyhanlp import *
from cilin import CilinSim
random.seed(1)


# 在 https://github.com/jasonwei20/eda_nlp 基础上修改。改为中文生成，同时将代码中不合理的地方进行修改。
# 其他的修改思路：  在随机替换和随机插入时，选择词频相对较低的词。


class EDAnlp:
    def __init__(self, stopwords_path="stopwords.txt"):
        self.__load_stopwords(stopwords_path)
        self.tokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")

    def __load_stopwords(self, stopwords_path):
        # stop words
        self.stop_words = set()
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for w in f:
                self.stop_words.add(w.strip())

    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=8, cut_tool='hanlp'):
        """
        **Synonym Replacement (SR):** 随机替换n个非停用词为同义词
        **Random Insertion (RI):** 随机选择一个非停用词的同义词，随机插入句子的某一个位置。重复n次。
        **Random Swap (RS):** 随机交换句子中的两个词，重复n次。
        **Random Deletion (RD):** 句子中每个词，以p的概率被删除。

        alpha_sr：随机替换比例
        num_aug：大于1时表示增强的数量。有四种方法，设置为4的倍数。默认至少一个。
                 小于1时，表示四种方法执行的概率，每种方法最多得到一个样本。仅适用于单句样本，不适用于句子对样本。
        cut_tool: hanlp or jieba

        return:
            list of string -- 每个词按空格分隔。
        """
        sentence = clean_sentence(sentence)
        if cut_tool == 'hanlp':
            words = [str(w.word) for w in self.tokenizer.segment(sentence)]
        elif cut_tool == 'jieba':
            words = [word for word in jieba.cut(sentence)]
        num_words = len(words)

        augmented_sentences = []
        augmented_sentences.append(' '.join(words))

        num_new_per_technique = int(num_aug / 4) + 1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # Synonym Replacement
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr, self.stop_words)
            augmented_sentences.append(' '.join(a_words))
        # Random Insertion
        for _ in range(num_new_per_technique):
            if len(words) < 3:
                continue
            a_words = random_insertion(words, n_ri, self.stop_words)
            augmented_sentences.append(' '.join(a_words))
        # Random Swap
        for _ in range(num_new_per_technique):
            if len(words) < 3:
                continue
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))
        # Random Deletion
        for _ in range(num_new_per_technique):
            if len(words) < 3:
                continue
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

        shuffle(augmented_sentences)
        # trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug + 1]
        else:
            keep_prob = num_aug
            augmented_sentences = [
                s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        return augmented_sentences


def clean_sentence(sent):
    sent = sent.replace('\n', '').replace('\t', '').replace('\u3000', '')
    # 删除对断句没有帮助的符号
    pattern_1 = re.compile(
        r"\(|\)|（|）|\"|“|”|\*|《|》|<|>|&|#|~|·|`|=|\+|\}|\{|\||、|｛|｝|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〿|–—|…|‧|﹏|")
    sent = re.sub(pattern_1, "", sent)
    # 断句符号统一为中文符号
    sent = re.sub(r"!", "！", sent)
    sent = re.sub(r"\?", "？", sent)
    sent = re.sub(r";", "；", sent)
    sent = re.sub(r",", "，", sent)
    # 去除网站，图片引用
    sent = re.sub(r"[！a-zA-z]+://[^\s]*", "", sent)
    # 去除邮箱地址
    sent = re.sub(r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", "", sent)

    sent = re.sub(r"@", "", sent)
    sent = sent.replace(' ', '').lower()
    return sent


def synonym_replacement(words, n, stop_words):
    "Replace n words in the sentence with synonyms"
    new_words = []
    random_word_list = list(
        set([word for word in words if word not in stop_words and word not in '，。？！']))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return new_words


def get_synonyms(word):
    "同义词选择方法一：synonyms"
    synonyms_cadidate = set()
    for sy_word in sy.nearby(word)[0]:
        synonyms_cadidate.add(sy_word)
    if word in synonyms_cadidate:
        synonyms_cadidate.remove(word)
    return synonyms_cadidate


synonym_handler =  CilinSim()
def get_synonyms_cilin(word):
    "同义词选择方法二：哈工大词林字典"
    synonyms = set()
    if word not in synonym_handler.vocab:
        print(word, '未被词林词林收录！')
    else:
        codes = synonym_handler.word_code[word]
        for code in codes:
            words = synonym_handler.code_word[code]
            synonyms.update(words)
    return synonyms


def random_deletion(words, p):
    "Randomly delete words from the sentence with probability p"
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def random_swap(words, n):
    "Randomly swap two words in the sentence n times"
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_insertion(words, n, stop_words):
    "Randomly insert n words into the sentence"
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, stop_words)
    return new_words


def add_word(new_words, stop_words):
    synonyms = []
    counter = 0
    random_word_list = list(
        set([word for word in new_words if word not in stop_words and word not in '，。？！']))
    while len(synonyms) < 1:
        random_word = random_word_list[random.randint(0, len(random_word_list) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = list(synonyms)[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


if __name__ == '__main__':
    s = """多知网6月25日消息，多知网第十二期OpenTalk活动上，AI新媒体《量子位》创始人孟鸿分享了从AI垂直领域的视角观察人工智能的内容。"""
    # print(clean_sentence(s))
    eda = EDAnlp()
    for sent in eda.eda(s):
        print('>>>', ''.join(sent))