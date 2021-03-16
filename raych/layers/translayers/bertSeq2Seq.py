# -*- coding:utf-8 -*-

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from overrides import overrides
from .bertmodel import BertModel, PreTrainedBertModel
from .bertpretrian import BertPreTrainingHeads
from .baselayers import ACT2FN


class BertModelIncr(BertModel):
    "attention 时，在key 和 value 中，融合history·信息"

    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    @overrides
    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None,
                prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True, task_idx=None):
        extended_attention_mask = self.get_extended_attention_mask(input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, task_idx=task_idx)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers)
        # last layer output
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return embedding_output, encoded_layers, pooled_output


class BertPreTrainingPairTransform(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingPairTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, pair_x, pair_y):
        hidden_states = torch.cat([pair_x, pair_y], dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainingPairRel(nn.Module):
    def __init__(self, config, num_rel=0):
        super(BertPreTrainingPairRel, self).__init__()
        self.R_xy = BertPreTrainingPairTransform(config)
        self.rel_emb = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, pair_x, pair_y, pair_r, pair_pos_neg_mask):
        # (batch, num_pair, hidden)
        xy = self.R_xy(pair_x, pair_y)
        r = self.rel_emb(pair_r)
        _batch, _num_pair, _hidden = xy.size()
        pair_score = (xy * r).sum(-1)
        # torch.bmm(xy.view(-1, 1, _hidden),r.view(-1, _hidden, 1)).view(_batch, _num_pair)
        # .mul_(-1.0): objective to loss
        return F.logsigmoid(pair_score * pair_pos_neg_mask.type_as(pair_score)).mul_(-1.0)


class BertForSeq2SeqDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config,
                 mask_word_id=4,
                 search_beam_size=1,
                 length_penalty=1.0,
                 eos_id=3,
                 sos_id=2,
                 forbid_duplicate_ngrams=False,
                 forbid_ignore_set=None,
                 ngram_size=3,
                 min_len=0):
        """

        :param config: Bertconfig
        :param mask_word_id: mask language model 使用的 mask id
        :param search_beam_size: beam_search 的 K 值
        :param length_penalty: beam search 参数
        :param eos_id: 结束 token 的 id
        :param sos_id: 开始 token 的 id
        :param forbid_duplicate_ngrams: 是否禁止出现重复 ngram
        :param forbid_ignore_set: ngram 白名单
        :param ngram_size: ngram
        :param min_len: 最小生成文本长度
        """
        super(BertForSeq2SeqDecoder, self).__init__(config)
        self.bert = BertModelIncr(config)
        # 进行 mask word predict 任务，预测下一个输出的 token
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.mask_word_id = mask_word_id

        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty

        self.eos_id = eos_id
        self.sos_id = sos_id

        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set

        self.ngram_size = ngram_size
        self.min_len = min_len


    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None):
        """input_ids: text长度， token_type_ids、position_ids，attention_mask： 为text加上待生成文本设置的长度，

        attention_mask：[[1,1,1,1,0,0,0]
                         [1,1,1,1,0,0,0]
                         [1,1,1,1,0,0,0]
                         [1,1,1,1,0,0,0]
                         [1,1,1,1,1,0,0]
                         [1,1,1,1,1,1,0]
                         [1,1,1,1,1,1,1]]  text部分长度为 4， 生成长度为 3

        """
        if self.search_beam_size > 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask, task_idx=task_idx)

        input_shape = list(input_ids.size())
        output_shape = list(token_type_ids.size())

        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids  # [batch, text_len]
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)  # tensor([[4]]）   mask for predict

        # next_pos： input text 部分进行完整 self attention的. 推断预测为 text 的下一个位置.
        #            next_pos表示当前预测的位置
        next_pos = input_length

        while next_pos < output_length:
            # curr_ids： 第一轮之后变成了 上一步预测的 token id tensor， 第一轮为 input_ids
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)  # mask_ids for BertPreTrainingHeads


            ############################# Main Idea #########################
            # 第二轮开始只输入 要预测的 target 部分的前一个 id 和 mask id，之前计算的部分，通过 prev_embedding 和 prev_encoded_layers 进行传递
            ############################# Main Idea #########################


            curr_token_type_ids = token_type_ids[:, start_pos:next_pos+1]
            # attention_mask: [batch, i, j]   每个位置 j ，对所有的 i 维位置进行attention时，mask部分 i 维中的位置
            curr_attention_mask = attention_mask[:, start_pos:next_pos+1, :next_pos+1]
            curr_position_ids = position_ids[:, start_pos:next_pos+1]

            # [batch, text len, hidden],  12 * [batch, text len, hidden]
            new_embedding, new_encoded_layers, _ = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            last_hidden = new_encoded_layers[-1][:, -1:, :]  # 预测下一个token的输出  [batch, 1, hidden]
            prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)  # [batch, 1, vocab size]
            _, max_ids = torch.max(prediction_scores, dim=-1)  # torch.Size([batch, 1])
            output_ids.append(max_ids)


            if prev_embedding is None:
                prev_embedding = new_embedding[:, :-1, :]  # 不包括 预测位置的 bert embedding， torch.Size([1, 156, 768]) 第二轮变为 torch.Size([1, 157, 768])
            else:
                prev_embedding = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)  # 12 * torch.Size([1, 156, 768])  第二轮 # 12 * torch.Size([1, 157, 768])
            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :] for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1) for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids  # 当前预测的输出词 id
            next_pos += 1

        return torch.cat(output_ids, dim=1)

    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length

        K = self.search_beam_size

        total_scores = []  # 每个 beam 每个 step 的当前得分
        beam_masks = []    # 是否 生成到 eos token 的 mask 标记
        step_ids = []      # 每一 step 选择的 token ids
        step_back_ptrs = []# 上一步选择的 beam 分支的 idx
        partial_seqs = []  # 当前每个 beam 生成的 token id 序列
        forbid_word_mask = None   # 重复 ngram 出现的 token 在 vocab 中 mask 掉  [btach * beam, 1, vocab size]
        buf_matrix = None  # 计算forbid_word_mask

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            new_embedding, new_encoded_layers, _ = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding,
                          prev_encoded_layers=prev_encoded_layers)


            # top k beam 计算
            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)

            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            # 最短长度之前，不能输出 eos id
            if self.min_len and (next_pos - input_length + 1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)


            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])

                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)

                back_ptrs = torch.div(k_ids, K).long()  # 分别计算展开之后的 k score 最优的beam 来自之前的哪一个beam branch
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)

            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).type_as(kk_scores))
            total_scores.append(k_scores)

            def first_expand(x):
                "展开 beam size 个分支"
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                # 根据 ids 从 x 中选出 k 个 beam
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2

                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2

                if id_rank < x_rank:
                    ids = torch.reshape(ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])

                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y


            ##################### beam size inference #######################
            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :])
            else:
                prev_embedding = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)
                prev_embedding = select_beam_items(prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                prev_encoded_layers = [first_expand(x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)


            ### 处理重复 ngram ####
            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs


                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n - 1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not (
                                self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))


                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros( (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).to(input_ids.device)
                    else:
                        forbid_word_mask = None
            next_pos += 1


        # [(batch, beam)] * 生成部分最大长度
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]


        # back tracking： 到 eos 停止， 或者 到达最大长度
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)

            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0, self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)


        ##################### 处理输出 ######################
        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces