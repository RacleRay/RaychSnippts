import os
import pandas as pd
import torch
import torch.nn as nn
import raych
from sklearn import model_selection
from raych.module.nlp.bert_binary_cls import BERTBaseUncased
from raych.datamanager.bertdata import BERTBinaryClsDa
from raych.util.info import get_machine_info, filter_warnings
from raych.util.randseed import prepare_seed
from raych.util import logger
from raych.util.tools import download_and_uncompress
from raych.preprocess.nlp.imdbpp import process_imdb


epochs = 10
batch_size = 128
lr = 3e-5
warmup_steps = 100
device = "cuda"
randseed = 2020


def download():
    # download ################################################
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    path = download_and_uncompress(url, "/tmp/")
    process_imdb(path=path,
                 csv_save_path=os.path.join(path, "imbd.csv"))


def main():
    # prepare ################################################
    logger.info(get_machine_info())
    filter_warnings()
    prepare_seed(randseed)

    # data ###################################################
    raw_data = pd.read_csv("/tmp/aclImdb_v1.tar/aclImdb/imbd.csv", 
                           names=['review', 'sentiment']).fillna(" ")
    print(raw_data.head(5))
    df_train, df_valid = model_selection.train_test_split(
        raw_data,
        test_size=0.1,
        random_state=randseed,
        stratify=raw_data.sentiment.values
    )
    # df_train = df_train.reset_index(drop=True)
    # df_valid = df_valid.reset_index(drop=True)
    # For test
    df_train = df_train.reset_index(drop=True)[: 2000]
    df_valid = df_valid.reset_index(drop=True)[: 1000]

    train_dataset = BERTBinaryClsDa(
        text=df_train.review.values,
        target=df_train.sentiment.values
    )
    valid_dataset = BERTBinaryClsDa(
        text=df_valid.review.values,
        target=df_valid.sentiment.values
    )

    # Model ###################################################
    n_train_steps = int(len(df_train) / batch_size * epochs)
    model = BERTBaseUncased(num_train_steps=n_train_steps,
                            num_warmup_steps=warmup_steps)

    tf_callback = raych.callbacks.TensorBoardLogger(log_dir=".logs/")
    earlystop = raych.callbacks.EarlyStopping(
        monitor="valid_loss", model_path="./weights/model_early_stop.bin")

    # Train ###################################################
    # 选择使用 学习率查找
    # model.find_lr(
    #     train_dataset,
    #     show_plot=True,
    #     fp16=True,
    #     train_bs=batch_size,
    #     valid_bs=batch_size,
    #     method='linear',
    #     init_value=1e-7,
    #     final_value=10,
    # )

    # 训练
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=batch_size,
        valid_bs=batch_size,
        device=device,
        epochs=epochs,
        callbacks=[tf_callback, earlystop],
        fp16=True,
        best_metric='accuracy',
        enable_fgm=False,
        learn_rate=3e-5
    )

    # 选择使用 FGM
    # model.fit(
    #     train_dataset,
    #     valid_dataset=valid_dataset,
    #     train_bs=batch_size,
    #     device=device,
    #     epochs=epochs,
    #     callbacks=[tf_callback, earlystop],
    #     learn_rate=3e-5,
    #     fp16=True,
    #     best_metric='accuracy',
    #     enable_fgm=True,
    #     reload_model="./weights/best_1_model_epoch3.bin"
    # )

    model.save("./weights/model_last_epoch.bin")


if __name__ == '__main__':
    # download()
    main()
