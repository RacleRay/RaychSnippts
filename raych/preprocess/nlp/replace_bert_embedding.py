import os
import re
import sys
import torch
import numpy as np
from transformers import BertConfig, BertTokenizer, BertModel, AlbertModel
from raych.preprocess.nlp.read_embedding import get_embedding_matrix_and_vocab


def replace_bert_embeddings(pretrained_model_path,
                            new_model_path,
                            w2v_file,
                            model_type=None,
                            use_random_init=False,
                            embedding_dim=768):
    """
    这是对于脱敏的比赛数据，进行embedding自定义初始化的方法。不是脱敏的数据时，能直接用BERT的参数就直接用。
    替换BERT的embedding参数，保存到新的model参数。效果并不好，实验用一用。

    Args:
        pretrained_model_path : pretrained_model_path
        new_model_path : 替换embedding后的模型参数保存路径
        w2v_file : w2v_file
        model_type : BERT 模型名称
        use_random_init : 使用随机初始化的embedding
        embedding_dim : 随机初始化embedding时使用
    Returns:
        None
    """
    vocab_list, vector_list = get_embedding_matrix_and_vocab(
        w2v_file, include_special_tokens=False
    )

    # 加载预训练模型部分
    MODEL_CLASSES = {
        'bert': (BertConfig, BertModel, BertTokenizer),
        'albert': (BertConfig, AlbertModel, BertTokenizer),
    }

    tokenizer = MODEL_CLASSES[model_type][2].from_pretrained(
        pretrained_model_path
    )
    config_class, model_class, _ = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(pretrained_model_path)
    model = model_class.from_pretrained(pretrained_model_path, config=config)

    bert_embed_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()
    bert_vocab = tokenizer.get_vocab()
    # print(type(bert_vocab))
    # print(len(bert_vocab))
    # print(bert_vocab["[PAD]"])
    # print(bert_embed_matrix[bert_vocab["[PAD]"]])

    # 构建新的vocab
    new_vocab_list, new_vector_list = [], []
    # [PAD], [UNK], [CLS], [SEP], [MASK] 的embedding不变
    for w in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        new_vocab_list.append(w)
        new_vector_list.append(bert_embed_matrix[bert_vocab[w]])

    for w_, vec_ in zip(vocab_list, vector_list):
        if not re.search("[0-9]", w_):
            print("non indexed word: ", w_)
            new_vocab_list.append(w_)
            new_vector_list.append(bert_embed_matrix[bert_vocab[w_]])
        else:
            new_vocab_list.append(w_)
            if not use_random_init:
                new_vector_list.append(vec_)
            else:
                new_vector_list.append(
                    (np.random.randn(embedding_dim).astype(np.float32) * 0.2).tolist()
                )

    assert len(new_vocab_list) == len(new_vector_list)

    vocab_file = os.path.join(new_model_path, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for w in new_vocab_list:
            f.write(w + "\n")

    config.vocab_size = len(new_vocab_list)
    config.save_pretrained(new_model_path)

    model.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.FloatTensor(new_vector_list))
    model.save_pretrained(new_model_path)


if __name__ == "__main__":
    pretrained_model_path = "..."
    new_model_path = "..."
    w2v_file = "..."

    replace_bert_embeddings(pretrained_model_path,
                            new_model_path,
                            w2v_file,
                            model_type="bert",
                            use_random_init=False,
                            embedding_dim=768)