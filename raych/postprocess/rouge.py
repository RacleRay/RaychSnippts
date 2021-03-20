from rouge import Rouge
import jieba  # or other libraries


# import os
# import sys
# import pathlib
# abs_path = pathlib.Path(__file__).parent.absolute()
# sys.path.append(sys.path.append(abs_path))


class RougeEval():
    def __init__(self, path):
        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []
        self.hypos = []
        self.refs = []
        self._process()

    def _process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r') as test:
            for line in test:
                source, ref = line.strip().split('<sep>')
                ref = ''.join(list(jieba.cut(ref))).replace('。', '.')
                self.sources.append(source)
                self.refs.append(ref)
        print(f'Test set contains {len(self.sources)} samples.')

    def build_hypos(self, predict):
        """Generate hypos for the dataset.
        Args:
            predict (model): The predictor model.
        """
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 100 == 0:
                print("predicting ", count, " ... ")
            self.hypos.append(predict.predict(source.split()))

    def get_average(self):
        assert len(self.hypos) > 0, 'Build hypotheses first!'
        print('Calculating average rouge scores.')
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)

    def one_sample(self, hypo, ref):
        return self.rouge.get_scores(hypo, ref)[0]



# rouge_eval = RougeEval(test_data_path)
# predict = Model()  # you design
# rouge_eval.build_hypos(predict)
# result = rouge_eval.get_average()

# print('rouge1: ', result['rouge-1'])
# print('rouge2: ', result['rouge-2'])
# print('rougeL: ', result['rouge-l'])