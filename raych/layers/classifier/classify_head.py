import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, dropout_rate, input_dim=128, num_labels=2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)   # [batch_size, hidden_dim]
        return self.linear(x)


class MultiSampleClassifier(nn.Module):
    def __init__(self, dropout_rate, dropout_num, agg_method="avg", input_dim=128, num_labels=2):
        "agg_method: 'avg'平均，'sum'求和.   dropout_num: number of dropout layers"
        super(MultiSampleClassifier, self).__init__()
        self.agg_method = agg_method
        self.dropout_num = dropout_num

        self.linear = nn.Linear(input_dim, num_labels)
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(dropout_num)]
        )

    def forward(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)

            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits

        # 相加还是求平均？
        if self.agg_method == "avg":
            logits = logits / self.dropout_num

        return logits