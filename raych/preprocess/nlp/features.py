import string
import re
import numpy as np
import jieba
import jieba.posseg as pseg

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
import os
from gensim import matutils
from itertools import islice


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def strQ2B(ustring):
    # 全角 半角符号 转换
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281
                  and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


############################################################################################################
# lda topic

def get_lda_features(lda_model, document):
    # 基于bag of word 格式数据获取lda的特征
    topic_importances = lda_model.get_document_topics(document,
                                                      minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:, 1]



############################################################################################################
# bert vec

def get_pretrain_embedding(text, tokenizer, model, max_length, device):
    '''get bert embedding'''
    # 通过bert tokenizer 来处理数据， 然后使用bert model 获取bert embedding
    text_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_length,
        ad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids, attention_mask, token_type_ids = text_dict['input_ids'], text_dict['attention_mask'], text_dict['token_type_ids']

    _, res = model(input_ids.to(device),
                   attention_mask=attention_mask.to(device),
                   token_type_ids=token_type_ids.to(device))

    return res.detach().cpu().numpy()[0]


"""
可选特征，auto encoder生成embedding
"""

############################################################################################################
# word vec

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from .bin word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)



def get_embed_mat(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


class EmbedReplace:
    "选择 tfidf 权重较小的词，根据这些词与其他词的word vec的相似性排序，选择进行词替换"
    def __init__(self, sample_path, wv_path):
        samples = []
        with open(sample_path, 'r', encoding='utf8') as file:
            for line in file:
                samples.append(line.strip())
        self.samples = samples

        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

        if os.path.exists('saved/tfidf.model'):
            self.tfidf_model = TfidfModel.load('saved/tfidf.model')
            self.dct = Dictionary.load('saved/tfidf.dict')
            self.corpus = [self.dct.doc2bow(doc) for doc in self.samples]
        else:
            self.dct = Dictionary(self.samples)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.samples]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('saved/tfidf.dict')
            self.tfidf_model.save('saved/tfidf.model')
            self.vocab_size = len(self.dct.token2id)


    def vectorize(self, docs, vocab_size):
        '''
        Args:
            docs: bag-of-words format, iterable of iterable of (int, number)
            vocab_size (int) – Number of terms in the dictionary. X-axis of the resulting matrix.
        '''
        return matutils.corpus2dense(docs, vocab_size)


    def _extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):
        """find high TFIDF socore keywords

        Args:
            dct (Dictionary): gensim.corpora Dictionary
            tfidf (list of tfidf):  model[doc]  [(word index, tfidf)]
            threshold (float)
            topk(int): num of highest TFIDF socore
        Returns:
            (list): A list of keywords
        """
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
        return list(islice([dct[w] for w, score in tfidf if score > threshold], topk))


    def _replace(self, token_list, doc, percent=0.3):
        """replace token by another token which is similar in wordvector

        Args:
            token_list (list): reference token list
            doc (list): A reference represented by a word bag model
        Returns:
            (str):  new reference str
        """
        keywords = self._extract_keywords(self.dct, self.tfidf_model[doc])  # 关键词不替换
        num = int(len(token_list) * percent)

        new_tokens = token_list.copy()
        indexes = np.random.choice(len(token_list), num)
        for index in indexes:
            token = token_list[index]
            if isChinese(token) and token not in keywords and token in self.wv:
                new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

        return ' '.join(new_tokens)


    def generate_samples(self, write_path, opt="w"):
        """generate new samples file
        Args:
            write_path (str):  new samples file path
        """
        replaced = []
        count = 0
        for token_list, doc in zip(self.samples, self.corpus):
            count += 1
            if count % 100 == 0:
                print("Processing samples ", count, "...")

                with open(write_path, opt, encoding='utf8') as file:
                    for line in self.samples:
                        file.write(line)
                        file.write('\n')

                replaced = []
            replaced.append(self._replace(token_list, doc))



############################################################################################################
# 设计的feature，较少使用，可实验
def custom_embedding_features(data, label_embedding, model_name='w2v'):
    '''
    word2vec max/mean & word2vec n-gram(2, 3, 4) max/mean & label embedding max/mean

    inputs:
        data， input data, DataFrame
        label_embedding, all label embedding
        model_name, w2v means word2vec
    return:
        data, DataFrame
    '''
    print('generate w2v & fast label max/mean')
    # 首先在预训练的词向量中获取标签的词向量句子,每一行表示一个标签表示
    # 每一行表示一个标签的embedding
    # 计算label embedding 具体参见文档
    data[model_name + '_label_mean'] = data[model_name].progress_apply(
        lambda x: joint_label_embedding(x, label_embedding, method='mean'))
    data[model_name + '_label_max'] = data[model_name].progress_apply(
        lambda x: joint_label_embedding(x, label_embedding, method='max'))

    print('generate embedding max/mean')
    # 将embedding 进行max, mean聚合
    data[model_name + '_mean'] = data[model_name].progress_apply(
        lambda x: np.mean(np.array(x), axis=0))
    data[model_name + '_max'] = data[model_name].progress_apply(
        lambda x: np.max(np.array(x), axis=0))

    print('generate embedding window max/mean')
    # 滑窗处理embedding 然后聚合
    data[model_name + '_win_2_mean'] = data[model_name].progress_apply(
        lambda x: embedding_within_windows(x, 2, method='mean'))
    data[model_name + '_win_3_mean'] = data[model_name].progress_apply(
        lambda x: embedding_within_windows(x, 3, method='mean'))
    data[model_name + '_win_4_mean'] = data[model_name].progress_apply(
        lambda x: embedding_within_windows(x, 4, method='mean'))
    data[model_name + '_win_2_max'] = data[model_name].progress_apply(
        lambda x: embedding_within_windows(x, 2, method='max'))
    data[model_name + '_win_3_max'] = data[model_name].progress_apply(
        lambda x: embedding_within_windows(x, 3, method='max'))
    data[model_name + '_win_4_max'] = data[model_name].progress_apply(
        lambda x: embedding_within_windows(x, 4, method='max'))
    return data


def embedding_within_windows(embedding_matrix, window_size=2, method='mean'):
    '''
    划窗聚合 word embeddings
    '''
    result_list = []
    for k1 in range(len(embedding_matrix)):
        if int(k1 + window_size) > len(embedding_matrix):
            result_list.extend(np.mean(embedding_matrix[k1:], axis=0).reshape(1, 300))
        else:
            result_list.extend(np.mean(embedding_matrix[k1:k1 + window_size], axis=0).reshape(1, 300))
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def softmax(x):
    x = x - np.max(x, axis=-1)
    return np.exp(x) / np.exp(x).sum(axis=-1)


def joint_label_embedding(example_matrix, label_embedding, method='mean'):
    '''
    论文《Joint embedding of words and labels for Text Classification》获取标签空间的词嵌入, numpy for feature generation.
    inputs:
        example_matrix(np.array 2D): denotes words embedding of input
        label_embedding(np.array 2D): denotes the embedding of all label
    return:
        the embedding by join label and word
    '''
    # 根据矩阵乘法来计算label与word之间的相似度
    # (len example, len label)
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (np.linalg.norm(example_matrix) * np.linalg.norm(label_embedding))

    # “类别-词语”的注意力机制
    attention = similarity_matrix.max(axis=1)
    # attention = similarity_matrix.mean(axis=1)
    attention = softmax(attention)
    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * attention
    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)



############################################################################################################
# Basic info

def tag_part_of_speech(data):
    # 获取文本的词性， 并计算名词，动词， 形容词的个数
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len([w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
    verb_count = len([w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
    return noun_count, adjective_count, verb_count


def query_cut(query):
    return list(jieba.cut(query))


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\s+", "", string)
    # string = re.sub(r"[^\u4e00-\u9fff]", "", string)
    string = re.sub(r"[^\u4e00-\u9fa5^.^,^!^?^:^;^、^a-z^A-Z^0-9]", "", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    return string.strip()


def get_basic_feature(df):
    '''
    处理pandas dataframe格式的输入，统计一些基本信息
    '''
    # 分字
    df['queryCut'] = df['queryCut'].progress_apply(lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in query_cut(x)])
    # 文本的长度
    df['length'] = df['queryCut'].progress_apply(lambda x: len(x))


    # 大写的个数
    df['capitals'] = df['queryCut'].progress_apply(lambda x: sum(1 for c in x if c.isupper()))
    # 大写 与 文本长度的占比
    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals']) / float(row['length']), axis=1)


    # 感叹号的个数
    df['num_exclamation_marks'] = df['queryCut'].progress_apply(lambda x: x.count('!'))
    # 问号个数
    df['num_question_marks'] = df['queryCut'].progress_apply(lambda x: x.count('?'))
    # 标点符号个数
    df['num_punctuation'] = df['queryCut'].progress_apply(lambda x: sum(x.count(w) for w in string.punctuation))
    # *&$%字符的个数
    df['num_symbols'] = df['queryCut'].progress_apply(lambda x: sum(x.count(w) for w in '*&$%'))


    # 词的个数
    df['num_words'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 唯一词的个数
    df['num_unique_words'] = df['queryCut'].progress_apply(lambda x: len(set(w for w in x)))
    # 唯一词 与总词数的比例
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']


    # 获取名词， 形容词， 动词的个数
    df['nouns'], df['adjectives'], df['verbs'] = zip(*df['text'].progress_apply(lambda x: tag_part_of_speech(x)))
    # 名词占总长度的比率
    df['nouns_vs_length'] = df['nouns'] / df['length']
    # 形容词占总长度的比率
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    # 动词占总长度的比率
    df['verbs_vs_length'] = df['verbs'] / df['length']
    # 名词占总词数的比率
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    # 形容词占总词数的比率
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    # 动词占总词数的比率
    df['verbs_vs_words'] = df['verbs'] / df['num_words']


    # 首字母大写其他小写的个数
    df["count_words_title"] = df["queryCut"].progress_apply(lambda x: len([w for w in x if w.istitle()]))

    # 平均词长度
    df["mean_word_len"] = df["text"].progress_apply(lambda x: np.mean([len(w) for w in x]))

    # 标点符号的占比
    df['punct_percent'] = df['num_punctuation'] * 100 / df['num_words']

    return df