


class CilinSim(object):
    """基于哈工大同义词词林扩展版处理同义词"""

    def __init__(self):
        """
        'code_word' 以编码为key，单词list为value的dict，一个编码有多个单词
        'word_code' 以单词为key，编码为value的dict，一个单词可能有多个编码
        'vocab' 所有不重复的单词，便于统计词汇总数。
        """
        self.code_word = {}
        self.word_code = {}
        self.vocab = set()
        self.file = './new_cilin.txt'
        self.read_cilin()

    def read_cilin(self):
        """读入同义词词林"""
        with open(self.file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                res = line.split()
                code = res[0]    # 词义编码
                words = res[1:]  # 同组的多个词
                self.vocab.update(words)
                self.code_word[code] = words

                for w in words:
                    if w in self.word_code.keys():
                        self.word_code[w].append(code)
                    else:
                        self.word_code[w] = [code]
                if len(code) < 6: # 要求的词类划分等级
                    continue