import torch
import transformers


class BERTBinaryClsDa:
    def __init__(self, text, target=None, tokenizer_name="bert-base-uncased", max_len=64):
        """BERT模型输入数据处理

        Arguments:
            text {Iterable[str]} -- 储存文本数据的容器对象
            target {Iterable[int]} -- 储存标签的容器对象

        Keyword Arguments:
            tokenizer_name {str} -- tokenizer name，可输入自行下载的预训练模型配套的文件路径
                 (default: {"bert-base-uncased"})
            max_len {int} -- 最大序列长度 (default: {64})
        """
        self.text = text
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        # text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        if any(self.target):
            label = self.target[item]
        else:  # test time
            label = None

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(label, dtype=torch.float),
        }
