from raych.callbacks.cbbase import Callback
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(Callback):
    def __init__(self, log_dir=".logs/"):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    def on_train_epoch_end(self, model):
        "每个 epoch 的训练过程记录"
        for metric in model.metrics["train"]:
            self.writer.add_scalar(f"train/{metric}",
                                   model.metrics["train"][metric], model.epoch)

    def on_valid_epoch_end(self, model):
        "每个 epoch 的验证过程记录"
        for metric in model.metrics["valid"]:
            self.writer.add_scalar(f"valid/{metric}",
                                   model.metrics["valid"][metric], model.epoch)

    def on_train_step_end(self, model):
        "整个训练过程，loss记录"
        self.writer.add_scalar(
            "train/step_loss",
            model.step_loss,
            model.train_step + model.epoch * model.total_train_step_epoch)

    def on_valid_step_end(self, model):
        "整个验证过程，loss记录"
        self.writer.add_scalar(
            "valid/step_loss",
            model.step_loss,
            model.valid_step + model.epoch * model.total_valid_step_epoch)
