
class ExponentialMovingAverage:
    def __init__(self, model, decay):
        """指数移动平均值

        Args:
            model (torch.Model): model
            decay (float): decay一般设置 0.999或0.9999 等接近1的数，这样 指数移动平均值 会更
                           趋于稳定。
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        "记录模型参数"
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, global_step=None):
        "更新指数移动平均参数值"
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow

                if global_step is not None:
                    decay = min(self.decay, (1 + global_step) / (10 + global_step))
                else:
                    decay = self.decay

                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        "将指数移动平均参数值复制到训练中的模型上"
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        "参数回滚"
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}