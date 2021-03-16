#-*- coding:utf-8 -*-

import os
import re
import torch
from transformers import BertTokenizer
from raych.layers.translayers.bertSeq2Seq import BertForSeq2SeqDecoder
from raych.layers.translayers.bertconfig import BertConfig
from raych.preprocess.nlp.detokenize import detokenize


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class TextCleaner:
    @classmethod
    def clean(cls, text: str) -> str:
        # 全角转换

        # urlの削除
        text = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" , text)

        # 英文数字半角
        replaces = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz;:/."
        text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94) if chr(0x21+i) in replaces}))

        text = re.sub(r"[\r\n]", " ", text)
        text = re.sub(r"[\u3000 \t]", " ", text)

        # 把两个以上的空格变成一个
        text = re.sub(r"\s+", " ", text)

        return text


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.word_subsample_prb = None
        self.sp_prob = None
        self.pieces_dir = None
        self.vocab_words = None
        self.pieces_threshold = 10
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128,
                 mode="s2s", source_type_id=0, target_type_id=1,
                 cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]'):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.task_idx = 3   # relax projection layer for different tasks, 输出hidden state的不同位置，进行不同的任务
                            # 只有当 config 中设置了 大于1 的 'relax_projection' 参数，task_idx 才生效

        self.mode = mode
        self.max_tgt_length = max_tgt_length

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token

        self.source_type_id = source_type_id
        self.target_type_id = target_type_id

        self.cc = 0

    def __call__(self, instance):
        tokens_a, max_a_len = instance

        # tokens
        padded_tokens_a = [self.cls_token] + tokens_a + [self.sep_token]
        assert len(padded_tokens_a) <= max_a_len + 2

        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += [self.pad_token] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2

        # 处理的是 source seq 与 target seq 拼接在一起的序列
        max_len_in_batch = min(self.max_tgt_length + max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        segment_ids = [self.source_type_id] * (len(padded_tokens_a)) + [self.target_type_id] * (max_len_in_batch - len(padded_tokens_a))

        # position_ids
        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing： use function from token to token index
        input_ids = self.indexer(tokens)

        self.cc += 1

        # Zero Padding
        input_mask = torch.zeros(max_len_in_batch, max_len_in_batch, dtype=torch.long)

        # for seq2seq source mask
        input_mask[:, :len(tokens_a)+2].fill_(1)

        # for seq2seq target mask
        second_st, second_end = len(padded_tokens_a), max_len_in_batch
        input_mask[second_st:second_end, second_st:second_end].copy_(self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # 最后要得到的 attention mask 效果
        # attention_mask：[[1,1,1,1,0,0,0]
        #                  [1,1,1,1,0,0,0]
        #                  [1,1,1,1,0,0,0]
        #                  [1,1,1,1,0,0,0]
        #                  [1,1,1,1,1,0,0]
        #                  [1,1,1,1,1,1,0]
        #                  [1,1,1,1,1,1,1]]  text部分长度为 4， 生成长度为 3

        return (input_ids, segment_ids, position_ids, input_mask, self.task_idx)


################################################################
################# Model
################################################################

class Seq2SeqBert:
    def __init__(self, pretrain_path, beam_size=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.beam_size = beam_size
        length_penalty = 0

        max_seq_length = 512
        max_tgt_length = 48
        ngram_size = 3
        min_len = 1
        forbid_duplicate_ngrams = True
        do_lower_case = True


        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path, do_lower_case=do_lower_case)
        self.tokenizer.max_len = max_seq_length
        self.text_cleaner = TextCleaner  # 自定义文本清洗部分


        config_file = os.path.join(pretrain_path, "config.json")
        print(f"load config : {config_file}")
        config = BertConfig.from_json_file(config_file)


        self.proc = Preprocess4Seq2seqDecoder(
            list(self.tokenizer.vocab.keys()),
            self.tokenizer.convert_tokens_to_ids,
            max_seq_length,
            max_tgt_length=max_tgt_length,
            source_type_id=config.source_type_id,
            target_type_id=config.target_type_id,
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            pad_token=self.tokenizer.pad_token
        )


        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.sep_token])

        # ngram 白名单
        forbid_ignore_set = None

        self.model = BertForSeq2SeqDecoder.from_pretrained(
            pretrain_path, config=config, mask_word_id=mask_word_id, search_beam_size=self.beam_size,
            length_penalty=length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=ngram_size, min_len=min_len,
            max_position_embeddings=max_seq_length,
        )

        self.model.to(self.device)
        self.model.eval()

        self.max_src_length = max_seq_length - 2 - max_tgt_length


    def generate(self, text):
        title = None


        try:
            source_text = self.text_cleaner.clean(text)
        except:
            return False, title

        if len(source_text) == 0:
            return False, title


        try:
            source_tokens = self.tokenizer.tokenize(source_text)[:self.max_src_length]
            tokenized_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

            max_a_len = len(source_tokens)  # tokens_len
            # input: [start ... end] tokens_len+2
            # segment_ids: [0] * tokens_len+2 + [1] * target_len
            # position_ids: pos tokens_len+2+target_len
            # input_mask： [max_src_length + max_tgt_length, max_src_length + max_tgt_length]
            # input_ids, segment_ids, position_ids, input_mask, self.task_idx
            instances = self.proc((source_tokens, max_a_len))

            with torch.no_grad():
                batch = batch_list_to_batch_tensors([instances])  # 5, batch_size, max_a_len
                batch = [t.to(self.device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, task_idx = batch

                # --- inference ---
                # dict : 'pred_seq', 'scores', 'wids', 'ptrs'
                traces = self.model(input_ids, token_type_ids, position_ids, input_mask, task_idx=task_idx)

                # --- postprocess ---
                if self.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()

                w_ids = output_ids[0]
                output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)

                output_tokens = []
                for t in output_buf:
                    if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                        break
                    output_tokens.append(t)

                title = ''.join(detokenize(output_tokens))

            return True, title
        except Exception as e:
            # print(e)
            return False, title


if __name__ == "__main__":
    pretrain_path = ""
    beam_size = 5

    generator = Seq2SeqBert(pretrain_path, beam_size)

    text = ""
    out = generator.generate(text)