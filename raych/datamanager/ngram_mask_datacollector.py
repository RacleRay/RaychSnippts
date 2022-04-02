import re
import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from raych.util.hanzi import punctuation  # For chinese characters


class BertDataset(Dataset):
    "Without labels, this is for pretrain use."
    def __init__(self, pretrain_csv_path, tokenizer: BertTokenizer):
        super(Dataset, self).__init__()
        self.data_dict = self._read_data(pretrain_csv_path, tokenizer)

    def _read_data(self, pretrain_csv_path, tokenizer: BertTokenizer) -> dict:
        pretrain_df = pd.read_csv(pretrain_csv_path, header=None, sep='\t')
        inputs = defaultdict(list)
        for i, row in tqdm(pretrain_df.iterrows(), desc='', total=len(pretrain_df)):
            sentence = row[0].strip()
            sentence = re.sub(r"[%s]+" % punctuation, '[SEP]', sentence)
            inputs_dict = tokenizer.encode_plus(sentence,
                                                add_special_tokens=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            inputs['input_ids'].append(inputs_dict['input_ids'])
            inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
            inputs['attention_mask'].append(inputs_dict['attention_mask'])
        return inputs

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class NGramDataCollator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def _pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        "Normalizing a batch to uniformed length."
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _ngram_mask(self, input_ids, max_seq_len):
        """按字进行ngram mask，其中选择不同 ngram 的概率可以调整。在google的一份代码中使用：
        15% use unigram , 20% use bigram, 30% use trigram, 20% use four gram, 15% use five gram.
        (https://github.com/LianxinRay/bert_wwm_ngram_masking_of_chinese/blob/master/ngram_dataset_create.py)
        但是google这份代码，并不是“真正的ngram”，而是在 ngram 的几个字中，依次遍历判断是否mask单个字，而不是看作一个整体。
        和这里的实现方式完全不同。
        """

        # 根据mask token的概率，计算需要mask的ngram数量
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))

        # 去除special token，初始化 ngram 结果的列表:  [[123], [412], [321], ...]
        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids: continue
            cand_indexes.append([i])

        # ngram长度。注意这里是按字为单位进行ngram的，所以 max_ngram 可以尝试更大一些。
        # 按词为单位需要进行预处理，复杂一些，暂时不考虑了，也没找到开源实现。
        # transformers库中的 transformers/src/transformers/data/data_collator.py
        # 有 DataCollatorForWholeWordMask，需要输入 chinese_ref 来标记哪些字为一个词。
        if len(input_ids) <= 32:
            max_ngram = 2
        else:
            max_ngram = 3
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)  # [1, 2, 3]
        pvals = 1. / np.arange(1, max_ngram + 1)  # [1, 0.5, 0.33]
        pvals /= pvals.sum(keepdims=True)  # [0.545, 0.273, 0.182]  可调


        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx: idx + n])  # cand_indexes: [ [123], [412] ]
                                                                # ngram_index: [ [[123]], [[123], [412]] ]
            ngram_indexes.append(ngram_index)  # ngram_indexes: [ [ [[123]],  [[123], [412]],  [[123], [412], [321]] ],  ... ]
        # shuffle according to the first token index.
        np.random.shuffle(ngram_indexes)


        covered_indexes = set()
        for cand_index_set in ngram_indexes: # cand_index_set: [ [[123]],  [[123], [412]],  [[123], [412], [321]] ]
            if len(covered_indexes) >= num_to_predict: break
            if not cand_index_set: continue

            # check the first token of ngram if it is duplicated.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue

            # random choose n according to pvals probability distribution.
            n = np.random.choice(ngrams, p=pvals)
            index_set = sum(cand_index_set[n - 1], [])  # [ [123], [412], [321] ]  if 3 gram

            # deal with limit
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0: break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue

            # cheak if there is any duplicate index
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue

            # mark the legal ngram index
            for index in index_set:
                covered_indexes.add(index)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))

        return torch.tensor(mask_labels[: max_seq_len])

    def ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """在 ngram 基础上，增加 StructBert 中，重构单词顺序的任务。
        其它 datacollector 可参考 https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
        """
        labels = inputs.clone()
        ngram_mask_matrix = mask_labels

        bs = inputs.shape[0]
        # word struct prediction
        for i in range(bs):
            now_input = inputs[i]
            now_input = now_input.cpu().numpy().tolist()

            now_probability_matrix = ngram_mask_matrix[i]
            now_probability_matrix = now_probability_matrix.cpu().numpy().tolist()

            for j in range(len(now_input)):
                if now_input[j] == self.tokenizer.sep_token_id:
                    sep_index = j

            # skip cls_ids, sep_ids, pad_ids
            choose_range = now_input[1: sep_index - 2]
            if len(choose_range) == 0:
                choose_range = now_input[1: 5]  # min length

            # random choose index to apply words order shuffle
            rd_token = np.random.choice(choose_range)
            token_idx = now_input.index(rd_token)

            # words order shuffled
            tmp = now_input[token_idx: token_idx + 3]
            np.random.shuffle(tmp)

            # change data and mark
            now_input[token_idx: token_idx + 3] = tmp
            now_probability_matrix[token_idx: token_idx + 3] = [1, 1, 1]

            inputs[i] = torch.tensor(now_input)
            ngram_mask_matrix[i] = torch.tensor(now_probability_matrix)

        # set special tokens` values in ngram_mask_matrix to 0
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        ngram_mask_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = ngram_mask_matrix.bool()

        # only reserve the ngram_mask labels, others all set to be -100. since we only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% mask will be reserved, and set to "mask" token id.
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # the rest 20% mask will be randomly changed to a random token id by a probability of 50%.
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self._pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


# if __name__ == "__main__":
    ### example for pretrain
    # tokenizer = BertTokenizer.from_pretrained(vocab_path)
    # model_config = BertConfig.from_pretrained(pretrain_model_path)

    # dataset = BertDataset(pretrain_data_path, tokenizer)
    # data_collator = NGramDataCollator(max_seq_len=seq_length,
    #                                tokenizer=tokenizer,
    #                                mlm_probability=mlm_probability)

    # model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=pretrain_model_path,
    #                                         config=model_config)
    # model.resize_token_embeddings(tokenizer.vocab_size)

    # training_args = TrainingArguments(...)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     data_collator=data_collator
    # )

    # trainer.train()
    # trainer.save_model(save_path)
    # tokenizer.save_pretrained(save_path)

