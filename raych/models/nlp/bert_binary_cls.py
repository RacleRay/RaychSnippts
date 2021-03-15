
import torch
import torch.nn as nn
import transformers
import raych
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from overrides import overrides


class BERTBaseUncased(raych.Pipeline):
    def __init__(self, *args, **kwargs):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 1)
        self.step_scheduler_after = "batch"

        self.num_warmup_steps = kwargs['num_warmup_steps']
        self.num_train_steps = kwargs['num_train_steps']

    @overrides
    def compute_loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.reshape(-1, 1))

    @overrides
    def forward(self, ids, mask, token_type_ids, targets=None):
        last_hidden_state, pooler_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        out = self.bert_drop(pooler_output)
        out = self.linear(out)
        loss = self.compute_loss(out, targets)
        acc = self.monitor_metrics(out, targets)
        return out, loss, acc

    @overrides
    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    @overrides
    def custom_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for name, p in param_optimizer if not any(nd in name for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for name, p in param_optimizer if any(nd in name for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.lr)
        return opt

    @overrides
    def custom_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_train_steps
        )
        return scheduler


if __name__ == "__main__":
    pass