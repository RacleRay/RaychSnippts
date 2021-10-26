import numpy as np
import torch
import torch.nn as nn
from raych.layers.position.position_embedding import SinusoidPositionalEmbedding
from raych.preprocess.nlp.read_embedding import get_embedding_matrix_and_vocab


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_dim, max_seq_len, dropout_rate=0.0, w2v_file="",
                 norm=True, freeze=False, return_position_embed=False):
        super(EmbeddingLayer, self).__init__()

        self.embed_dim = embed_dim
        self.norm = norm
        self.return_position_embed = return_position_embed

        # 加载embedding：从word2vec的文件中加载
        vocab_list, vector_list = get_embedding_matrix_and_vocab(
            w2v_file, skip_first_line=True, include_special_tokens=True)

        assert self.embed_dim == len(vector_list[0])
        assert len(vocab_list) == len(vector_list)

        self.vocab_list = vocab_list
        vector_list = np.asarray(vector_list)

        if len(w2v_file) == 0:
            self.word_embedding = nn.Embedding(
                len(self.vocab_list),
                self.embed_dim,
            )
        else:
            self.word_embedding = nn.Embedding(
                len(self.vocab_list),
                self.embed_dim,
            ).from_pretrained(torch.FloatTensor(vector_list), freeze=freeze)

        self.positional_embedding = SinusoidPositionalEmbedding(
            max_len=max_seq_len,
            embed_dim=self.embed_dim
        )

        # embedding dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.embed_dim)

    def forward(self, input_ids=None, position_ids=None, **kwargs):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device

        embeddings = self.word_embedding(input_ids)

        if self.return_position_embed:
            # 加上positional embedding
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
            position_embeddings = self.positional_embedding(position_ids)
            embeddings = embeddings + position_embeddings

        if self.norm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings