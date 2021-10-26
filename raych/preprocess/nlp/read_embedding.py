import numpy as np
import tqdm


def get_embedding_dict(fileName, skip_first_line=False):
    """
    Read Embedding Function

    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'rb') as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            if skip_first_line:
                if i == 0:
                    continue
            line_uni = line.strip()
            line_uni = line_uni.decode('utf-8')
            values = line_uni.split(' ')
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                print(values, len(values))
            embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix

    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) < 1:
        raise ValueError('Input dimension less than 1')

    vocab_sz = max(vocab_dic.values()) + 1
    EMBEDDING_DIM = len(list(embed_dic.values())[0])

    embedding_matrix = np.random.rand(vocab_sz, EMBEDDING_DIM).astype(np.float32) * 0.05
    valid_mask = np.ones(vocab_sz, dtype=np.bool)

    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be randomly initialized.
            embedding_matrix[i] = embedding_vector
        else:
            valid_mask[i] = False
    return embedding_matrix, valid_mask


def get_embedding_matrix_and_vocab(w2v_file, skip_first_line=True, include_special_tokens=True):
    """
    Construct vocab_list and vector_list

    Args:
        w2v_file : gensim训练的word vec file，或者其它相同格式的文件
        skip_first_line : 是否跳过第一行
        include_special_tokens： 加上 PAD 和 UNK
    Returns:
        vocab_list：词汇列表
        vector_list：词向量列表
    """
    embedding_dim = None

    vocab_list = []
    vector_list = []
    with open(w2v_file, "r", encoding="utf-8") as f_in:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            if skip_first_line:
                if i == 0:
                    continue

            line = line.strip()
            if not line: continue

            line = line.split(" ")
            w_ = line[0]
            vec_ = line[1: ]
            vec_ = [float(w.strip()) for w in vec_]

            if embedding_dim == None:
                embedding_dim = len(vec_)
            else:
                assert embedding_dim == len(vec_)

            vocab_list.append(w_)
            vector_list.append(vec_)

    # 添加两个特殊的字符： PAD 和 UNK
    if include_special_tokens:
        vocab_list = ["pad", "unk"] + vocab_list
        pad_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        unk_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        vector_list = [pad_vec_, unk_vec_] + vector_list

    return vocab_list, vector_list