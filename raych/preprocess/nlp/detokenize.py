#-*- coding:utf-8 -*-


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def sanitize_wordpiece(wordpiece: str) -> str:
    """
    Sanitizes wordpieces from BERT, RoBERTa or ALBERT tokenizers.
    """
    if wordpiece.startswith("##"):
        return wordpiece[2:]
    elif wordpiece.startswith("Ġ"):
        return wordpiece[1:]
    elif wordpiece.startswith("▁"):
        return wordpiece[1:]
    else:
        return wordpiece