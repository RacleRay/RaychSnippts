import os

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizer


def train_tokenizer(file_path, vocab_size, min_frequency, out_path):
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # 自定义 [unused{i}]，比如 [NEWTOKEN]。
    # 在使用新的tokenizer，或者增加了 special_token 后，加载预训练模型之后都应该增加如下一行：
    #      ## model.resize_token_embeddings(tokenizer.vocab_size)
    # 另外在使用 tokenizer时也可以增加新的token，在 vocab.txt 中替换 [unused*] 为自定义的token。如下：
    #      ## tokenizer = BertTokenizer.from_pretrained(pretrain_model_path,additional_special_tokens=added_token)

    # for i in range(100):
    #     special_tokens.append(f"[unused{i}]")

    tokenizer.train(
        files=[file_path, ],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        limit_alphabet=vocab_size,
        wordpieces_prefix="##"
    )
    os.makedirs(out_path, exist_ok=True)
    tokenizer.save_model(out_path)
    tokenizer = BertTokenizer.from_pretrained(out_path,
                                              do_lower_case=False,
                                              strip_accents=False)
    tokenizer.save_pretrained(out_path)

    print(f'Tokenizer saved, with vocab_size: {tokenizer.vocab_size}')