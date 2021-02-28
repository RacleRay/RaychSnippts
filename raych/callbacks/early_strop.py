import numpy as np
from raych.state import ModelState
from raych.callbacks.cbbase import Callback
from raych.util import logger


class EarlyStopping(Callback):
    def __init__(self,
                 monitor,
                 model_path="./weights/model_early_stop.bin",
                 patience=5,
                 mode="min",
                 delta=0.001):
        """Early stopping Callback

        Args:
            monitor (str): 设置早停的监控指标，"train_loss"表示按照train阶段的loss进行早停，
                必须以 train_ 或者 valid_ 开头，后接上 loss （内置），或则其他指标。其他指标
                需要在 Model.monitor_metrics 方法中计算，经 train_one_epoch 方法，保存在 self.metrics
                ```
                def monitor_metrics(self, outputs, targets):
                    acc = ...
                    return {"accuracy": accuracy}

                EarlyStopping(monitor="valid_accuracy", ...)
                # EarlyStopping(monitor="train_accuracy", ...)
                ```
            model_path (str): 保存路径， 比如 weight/model.bin
            patience (int, optional): 早停等待轮次. Defaults to 5.
            mode (str, optional): 指标最大化、还是最小化. Defaults to "min".
            delta (float, optional): score比较时的偏差容忍度. Defaults to 0.001.
        """
        self.monitor = monitor
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.model_path = model_path

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

        if self.monitor.startswith("train_"):
            self.model_state = "train"
            self.metric_name = self.monitor[len("train_"):]
        elif self.monitor.startswith("valid_"):
            self.model_state = "valid"
            self.metric_name = self.monitor[len("valid_"):]
        else:
            raise Exception("monitor must start with train_ or valid_")

    def on_epoch_end(self, model):
        "通过 CallbackRunner 对象，回调该函数"
        epoch_score = model.metrics[self.model_state][self.metric_name]
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(" [Epoch callback] EarlyStopping counter: {} out of {}".format(
                self.counter, self.patience))
            # 设置 主训练循环 的早停哨兵值
            if self.counter >= self.patience:
                model.model_state = ModelState.END
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            logger.info("[Epoch callback] Validation {} improved ({} --> {}). Saving model!".format(
                self.metric_name, self.val_score, epoch_score))
            model.save(self.model_path)
        self.val_score = epoch_score
