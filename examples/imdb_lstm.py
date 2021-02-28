import torch
import torch.nn as nn
import raych
from raych.datamanager.opendata import load_tt_data
from raych.util.info import get_machine_info, filter_warnings
from raych.util.randseed import prepare_seed
from raych.util import logger


class RNN(raych.Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()
        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths, targets=None):
        # text = [sent len, batch size]

        # [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        # 注意将text_lengths放在 cpu 上
        # pack sequence, 使得 rnn 只处理 非pad 位置。且返回的 hidden and cell 是从最后一个 非pad 位置输出
        #                而不是从序列最后一个 pad 输出
        packed_sequence = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            text_lengths.to(
                                                                'cpu'),
                                                            batch_first=True)
        # hidden: [num layers * num directions, batch size, hid dim]
        # cell  : [num layers * num directions, batch size, hid dim]
        packed_output, (hidden, cell) = self.rnn(packed_sequence)
        # unpack sequence： pad输出与输入的shape一致
        # output: [sent len, batch size, hid dim * num directions]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        # final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # hidden: [batch size, hid dim * num directions]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # 使用 raych.Model 需要计算
        out = self.fc(hidden)
        targets = targets.reshape(-1, self.output_dim)
        loss = self.compute_loss(out, targets)
        acc = self.monitor_metrics(out, targets)

        return out, loss, acc

    def model_fn(self, data):
        "BucketIterator输入为Batch对象，需要重写model_fn, 原model_fn只处理字典输入"
        text, text_lengths = data.text
        if 'label' in data.fields and data.label[0] != None:
            label = data.label - 1.0
        if self.fp16:
            with torch.cuda.amp.autocast():
                # forward
                out, loss, acc = self(text, text_lengths, label)
        else:
            out, loss, acc = self(text, text_lengths, label)
        return out, loss, acc

    def compute_loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        pred = torch.round(torch.sigmoid(outputs)).cpu().detach().numpy()
        targ = targets.cpu().detach().numpy()
        correct = (pred == targ)
        acc = correct.sum() / len(correct)
        return {"accuracy": acc}

    def custom_optimizer(self):
        param_optimizer=list(self.named_parameters())
        no_decay=["bias", "LayerNorm.bias"]
        optimizer_parameters=[
            {
                "params": [p for name, p in param_optimizer if not any(nd in name for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for name, p in param_optimizer if any(nd in name for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt= torch.optim.Adam(optimizer_parameters, lr=self.lr) # lr在fit时传入
        return opt


if __name__ == "__main__":
    # prepare ################################################
    logger.info(get_machine_info())
    filter_warnings()
    prepare_seed(23)

    batch_size= 64
    max_vocab_size= 25000
    max_seq_len= 128
    embed_dim= 100
    hid_dim= 256
    out_dim= 1
    n_layers= 2
    bidirectional= True
    dropout= 0.5
    device= "cuda"
    epochs= 10

    train_iter, test_iter, label_size, vocab_size, embeddings, pad_idx = load_tt_data('imdb',
                                                                                    batch_size,
                                                                                    max_seq_len,
                                                                                    max_size = max_vocab_size,
                                                                                    dim = embed_dim)

    model=RNN(vocab_size,
                embed_dim,
                hid_dim,
                out_dim,
                n_layers,
                bidirectional,
                dropout,
                pad_idx)
    model.embedding.weight.data.copy_(embeddings)

    tf_callback = raych.callbacks.TensorBoardLogger(log_dir=".logs/lstm/")
    earlystop = raych.callbacks.EarlyStopping(
        monitor="valid_loss", model_path="./weights/model_early_stop.bin")
    
    model.fit(
        train_iter,
        valid_dataset=test_iter,
        train_bs=batch_size,
        valid_bs=batch_size,
        device=device,
        epochs=epochs,
        callbacks=[tf_callback, earlystop],
        fp16=True,
        best_metric='accuracy',
        enable_fgm=False,
        learn_rate=0.001,
        is_dataloader=True
    )

    model.save("./weights/model_last_epoch.bin")
