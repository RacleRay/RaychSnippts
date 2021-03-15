from typing import List


class Callback:
    """自定义任意阶段的回调函数

    注意方法名称，和 TrainingState 中定义的属性要一致，通过 getattr 获取方法
    """
    def on_epoch_start(self, model, **kwargs):
        return

    def on_epoch_end(self, model, **kwargs):
        return

    def on_train_epoch_start(self, model, **kwargs):
        return

    def on_train_epoch_end(self, model, **kwargs):
        return

    def on_valid_epoch_start(self, model, **kwargs):
        return

    def on_valid_epoch_end(self, model, **kwargs):
        return

    def on_train_step_start(self, model, **kwargs):
        return

    def on_train_step_end(self, model, **kwargs):
        return

    def on_valid_step_start(self, model, **kwargs):
        return

    def on_valid_step_end(self, model, **kwargs):
        return

    def on_test_step_start(self, model, **kwargs):
        return

    def on_test_step_end(self, model, **kwargs):
        return

    def on_train_start(self, model, **kwargs):
        return

    def on_train_end(self, model, **kwargs):
        return


class CallbackRunner:
    def __init__(self, callbacks: List[Callback], model):
        """初始化 callback runner

        Args:
            callbacks (List[Callback]): 包含Callback对象的列表
            model (raych.Pipeline): 继承自nn.Module的自定义Model对象
        """
        self.model = model
        self.callbacks = callbacks

    def __call__(self, state, **kwargs):
        "指定 状态阶段，调用自定义的 callback 方法"
        for callback in self.callbacks:
            # 调用 与state 名称相同的 callback 对象成员方法
            _ = getattr(callback, state)(self.model, **kwargs)
