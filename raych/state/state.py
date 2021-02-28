class ModelState:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    END = "end"   # for early stop


class TrainingState:
    """根据以下不同的State值，Model会调用不同的自定义 hook callback runner"""
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"

    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"

    TRAIN_EPOCH_START = "on_train_epoch_start"
    TRAIN_EPOCH_END = "on_train_epoch_end"

    VALID_EPOCH_START = "on_valid_epoch_start"
    VALID_EPOCH_END = "on_valid_epoch_end"

    TRAIN_STEP_START = "on_train_step_start"
    TRAIN_STEP_END = "on_train_step_end"

    VALID_STEP_START = "on_valid_step_start"
    VALID_STEP_END = "on_valid_step_end"

    TEST_STEP_START = "on_test_step_start"
    TEST_STEP_END = "on_test_step_end"


class MetricMeter:
    """储存 metric 累计数据的数据结构"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count