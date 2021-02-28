import psutil
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from raych.callbacks.cbbase import CallbackRunner
from raych.state import TrainingState, ModelState, MetricMeter
from raych.util import logger
from raych.util.tools import create_dir, remove_file
from raych.util.heap import Heap, CheckpointStruct


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        # device
        self.device = None

        # optimizition
        self.optimizer = None
        self.scheduler = None
        self.step_scheduler_after = None

        # boost with half precision
        self.fp16 = False
        self.scaler = None

        # data loader
        self.train_loader = None
        self.valid_loader = None

        # progress
        self.epoch = 0
        self.train_step = 0
        self.valid_step = 0
        self.total_train_step_epoch = 0
        self.total_valid_step_epoch = 0
        self._model_state = None
        self._train_state = None

        # metric
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}

        # best scores (min heap)
        self.best_scores = None
        self.the_one = -np.inf

        # callback
        self.callback_runner = None

    def fit(self,
            train_dataset,
            valid_dataset=None,
            train_sampler=None,
            valid_sampler=None,
            shuffle=True,
            device="cuda",
            epochs=10,
            train_bs=16,
            valid_bs=16,
            n_jobs=4,
            callbacks=None,
            fp16=False,
            train_collate_fn=None,
            valid_collate_fn=None,
            enable_fgm=False,
            fgm_epsilon=1.0,
            emb_prefix='emb',
            metric_cmp='max',
            enable_bestk=False,
            k_best=3,
            enable_bestone=True,
            best_metric="loss",
            reload_model="",
            learn_rate=0.01,
            is_dataloader=False):
        """模型主训练逻辑

        Args:
            train_dataset (Dataset)
            valid_dataset (Dataset, optional): 验证集，可不输入. Defaults to None.
            train_sampler (Sampler, optional): 训练数据采样逻辑，与shuffle设置互斥.
                https://pytorch.org/docs/stable/data.html?highlight=sampler#torch.utils.data.Sampler
                Defaults to None.
            valid_sampler (Sampler, optional): 验证数据采样逻辑. 一般验证数据不用设置
                Defaults to None.
            shuffle (bool, optional): 打乱数据loader，设置 sampler 时，必须为 False.
                Defaults to True.
            device (str, optional): Defaults to "cuda".
            epochs (int, optional): 训练总轮次. Defaults to 10.
            train_bs (int, optional): train batch size. Defaults to 16.
            valid_bs (int, optional): valid batch size. Defaults to 16.
            n_jobs (int, optional): 处理数据loader时，并行处理任务数. Defaults to 4.
            callbacks (List[Callbacks], optional): 自定义的 callback 对象list. Defaults to None.
            fp16 (bool, optional): 半精度训练. Defaults to False.
            train_collate_fn (callable, optional): 从 字典 形式的输入数据集中，采样出一个batch数据的函数. Defaults to None.
            valid_collate_fn (callable, optional): 从 字典 形式的输入数据集中，采样出一个batch数据的函数. Defaults to None.
            enable_fgm (bool, optional): 是否使用 fgm 对抗训练算法，此处只在NLP模型中使用.
            fgm_epsilon (float, optional): fgm 中，修正梯度时，乘法系数项. Defaults to 1.0.
            emb_prefix (str, optional): fgm 中，指定embedding层名称前缀. Defaults to "emb".
            metric_cmp (str, optional): 'max'或者'min', 训练指标取最大，还是取最小. Defaults to "max".
            enable_bestk (bool, optional): 保存最好的 k 个模型，以valid指标为准. Defaults to False.
            k_best (int, optional): enable_bestk的 k . Defaults to 3.
            enable_bestone (bool, optional): 保存最好的模型，以valid指标为准，和EarlyStopping callback独立. 
                Defaults to True.
            best_metric (str, optional): 最好模型的 评价基准，默认 loss, 可设置为在 monitor_metrics 方法中自定义的名称. 
                Defaults to loss.
            reload_model (str, optional): 加载 reload_model 路径的 checkpoint 文件继续训练. Defaults to "".
            learn_rate (float, optional): 学习率. Defaults to 0.01.
            is_dataloader (bool, optional): 输入的 train_dataset，valid_dataset 是否已经是 DataLoader 对象. 
                Defaults to False.
        """
        self._init_model(device=device,
                         train_dataset=train_dataset,
                         valid_dataset=valid_dataset,
                         shuffle=shuffle,
                         train_sampler=train_sampler,
                         valid_sampler=valid_sampler,
                         train_bs=train_bs,
                         valid_bs=valid_bs,
                         n_jobs=n_jobs,
                         callbacks=callbacks,
                         fp16=fp16,
                         train_collate_fn=train_collate_fn,
                         valid_collate_fn=valid_collate_fn,
                         enable_fgm=enable_fgm,
                         fgm_epsilon=fgm_epsilon,
                         emb_prefix=emb_prefix,
                         reload_model=reload_model,
                         learn_rate=learn_rate,
                         is_dataloader=is_dataloader)

        for _ in range(epochs):
            self.epoch += 1

            # 调用 on_epoch_start 回调函数，每次设置train_state，触发回调函数调用逻辑
            self.train_state = TrainingState.EPOCH_START

            # train
            self.train_state = TrainingState.TRAIN_EPOCH_START
            _ = self.train_one_epoch(self.train_loader)
            self.train_state = TrainingState.TRAIN_EPOCH_END

            # valid
            if self.valid_loader:
                self.train_state = TrainingState.VALID_EPOCH_START
                _ = self.validate_one_epoch(self.valid_loader)
                self.train_state = TrainingState.VALID_EPOCH_END

            # keep best 3 model
            metric_val = self.metrics["valid"][best_metric]
            if metric_cmp == 'min':
                metric_val *= -1  # 转换为 最大 为目标
            if enable_bestk and (self.best_scores.heap[1].score < metric_val) :
                save_name = f"best_{k_best}_model_epoch{self.epoch}.bin"
                if enable_fgm:
                    save_name = f"best_{k_best}_model_epoch{self.epoch}_fgm.bin"
                record = CheckpointStruct(metric_val, './weights/' + save_name)
                if len(self.best_scores) >= k_best:
                    pop_record = self.best_scores.pop()
                    remove_file(pop_record.name)
                self.best_scores.push(record)
                self.save('./weights/' + save_name)

            # keep best 1 model
            if enable_bestone and (metric_val > self.the_one):
                self.the_one = metric_val
                if self.pre_best_path != "":
                    remove_file(self.pre_best_path)
                save_name = f"best_1_model_epoch{self.epoch}.bin"
                if enable_fgm:
                    save_name = f"best_1_model_epoch{self.epoch}_fgm.bin"
                self.save('./weights/' + save_name)
                self.pre_best_path = './weights/' + save_name

            # 每个epoch调用scheduler，学习率修改
            if self.scheduler and (self.step_scheduler_after == "epoch"):
                self.scheduler.step()

            self.train_state = TrainingState.EPOCH_END

            # early stop, when use early stop callback
            if self._model_state == "end":
                break

        self.train_state = TrainingState.TRAIN_END

    def find_lr(self,
                train_dataset,
                init_value=1e-8,
                final_value=10.0,
                method="linear",
                gamma=1.0,
                show_plot=False,
                device="cuda",
                epochs=1,
                train_bs=16,
                train_sampler=None,
                n_jobs=4,
                callbacks=None,
                fp16=False,
                train_collate_fn=None,
                valid_collate_fn=None,
                save_img_name="find_lr.png"):
        """简易最佳学习率寻找方法，找到loss下降最快的lr，设置为初始学习，可以逐步缩小搜索范围

        Args:
            train_dataset (Dataset)：
            init_value (float, optional): 学习率搜索下界, cos 搜索时应该缩小搜索范围. Defaults to 1e-8.
            final_value (float, optional): 学习率搜索上界, cos 搜索时应该缩小搜索范围，
                cos 搜索时，init_value与final_value一般3、4个数量级以内. Defaults to 10.0.
            method (str, optional): "linear"学习率均增加相同倍数，"cos"学习率按余弦函数增长. Defaults to "linear".
            gamma (float, optional): "cos"变化几个 pi 半周期，默认为 1. Defaults to 1.0.
            show_plot (bool, optional): 是否plot变化图像. Defaults to False.
            device (str, optional): 设备. Defaults to "cuda".
            epochs (int, optional): 搜索时一般设置为1. Defaults to 1.
            train_bs (int, optional): batch size for training. Defaults to 16.
            train_sampler (Sampler, optional): 训练数据采样逻辑，与shuffle设置互斥.
                Defaults to None.
            n_jobs (int, optional): 处理数据loader时，并行处理任务数. Defaults to 4.
            callbacks (List[Callbacks], optional): 自定义的 callback 对象list. Defaults to None.
            fp16 (bool, optional): 半精度训练. Defaults to False.
            train_collate_fn (callable, optional): 从 字典 形式的输入数据集中，采样出一个batch数据的函数. Defaults to None.
            valid_collate_fn (callable, optional): 从 字典 形式的输入数据集中，采样出一个batch数据的函数. Defaults to None.
        Returns:
            (list, list): (学习率列表，对应损失列表)
        """
        self._init_model(device=device,
                         train_dataset=train_dataset,
                         valid_dataset=None,
                         shuffle=True,
                         train_sampler=train_sampler,
                         valid_sampler=None,
                         train_bs=train_bs,
                         valid_bs=None,
                         n_jobs=n_jobs,
                         callbacks=callbacks,
                         fp16=fp16,
                         train_collate_fn=train_collate_fn,
                         valid_collate_fn=None)

        number_in_epoch = len(self.train_loader) - 1
        linear_step = (final_value / init_value)**(1 / number_in_epoch)

        lr = init_value
        self.optimizer.param_groups[0]["lr"] = lr

        best_loss = 10000.0
        batch_num = 0
        losses = []
        lrs = []
        for data in tqdm(self.train_loader):
            batch_num += 1
            self.optimizer.zero_grad()

            _, loss, _ = self.model_fn(data)

            # Crash out if loss explodes
            if batch_num > 1 and loss > 10 * best_loss:
                break

            # Record the best loss
            if loss < best_loss or batch_num == 1:
                best_loss = loss

            # Store the values
            losses.append(loss.item())
            if method == 'linear':
                lrs.append(math.log10(lr))
            elif method == 'cos':
                lrs.append(lr)

            # Do the backward pass and optimize
            with torch.set_grad_enabled(True):
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # Update the lr for the next step and store
            if method == 'linear':
                lr *= linear_step
            elif method == 'cos':
                lr = init_value + (final_value - init_value) * (1 + math.cos(
                    gamma * math.pi * batch_num / number_in_epoch)) / 2
            self.optimizer.param_groups[0]["lr"] = lr

        if (len(lrs) > 20):
            lrs, losses = lrs[10:-5], losses[10:-5]
        if show_plot:
            plt.plot(lrs, losses)
            plt.savefig(save_img_name)

        return lrs, losses

    def _init_model(self,
                    device,
                    train_dataset,
                    valid_dataset,
                    shuffle,
                    train_sampler,
                    valid_sampler,
                    train_bs,
                    valid_bs,
                    n_jobs,
                    callbacks,
                    fp16,
                    train_collate_fn,
                    valid_collate_fn,
                    find_lr=False,
                    enable_fgm=False,
                    fgm_epsilon=1.0,
                    emb_prefix='emb',
                    reload_model="",
                    learn_rate=0.01,
                    is_dataloader=False):

        create_dir('./weights')

        # init record heap
        self.best_scores = Heap([CheckpointStruct(-1, "")])
        self.pre_best_path = ""
        
        if callbacks is None:
            callbacks = list()

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if enable_fgm: self.backup = {}

        self.device = device
        self.enable_fgm = enable_fgm
        self.epsilon = fgm_epsilon
        self.emb_name = emb_prefix

        # 检查当前定义的模型是否在目标设备上
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        # 输入是 Dataset 对象
        if self.train_loader is None and not is_dataloader:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_bs,
                num_workers=n_jobs,
                sampler=train_sampler,
                shuffle=shuffle,
                collate_fn=train_collate_fn,
            )
            self.total_train_step_epoch = len(self.train_loader)
        if self.valid_loader is None and not is_dataloader:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=valid_bs,
                    num_workers=n_jobs,
                    sampler=valid_sampler,
                    shuffle=False,
                    collate_fn=valid_collate_fn,
                )
                self.total_valid_step_epoch = len(self.valid_loader)

        # 输入是 DataLoader 对象
        if is_dataloader and self.train_loader is None:
            self.train_loader = train_dataset
        if is_dataloader and self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = valid_dataset
                self.total_valid_step_epoch = len(self.valid_loader)

        self.lr = learn_rate
        if self.optimizer is None:
            self.optimizer = self.custom_optimizer()

        if self.scheduler is None and not find_lr:
            self.scheduler = self.custom_scheduler()

        # 半精度训练
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # 回调函数设置
        self.callback_runner = CallbackRunner(callbacks, self)

        # load model from checkpoint
        if reload_model != "":
            self.load(reload_model)

        self.train_state = TrainingState.TRAIN_START

    def train_one_epoch(self, data_loader):
        self.train()
        self.model_state = ModelState.TRAIN
        # 初始化 损失记录对象
        losses = MetricMeter()
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tqdm_loader):
            # trian step
            self.train_state = TrainingState.TRAIN_STEP_START
            _, self.step_loss, metrics = self.train_one_step(data)
            self.train_state = TrainingState.TRAIN_STEP_END
            losses.update(self.step_loss.item())

            # 初始化 评价指标记录对象
            if b_idx == 0:
                metrics_meter = {k: MetricMeter() for k in metrics}

            # 记录一个 epoch 累计平均评价指标
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m])
                monitor[m_m] = metrics_meter[m_m].avg

            self.train_step += 1
            tqdm_loader.set_postfix(loss=losses.avg, stage="train", **monitor)
        tqdm_loader.close()

        # 根据阶段，记录 指标和损失值
        self.metrics[self._model_state].update(monitor)
        self.metrics[self._model_state]["loss"] = losses.avg

        return losses.avg

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        # forward
        output, loss, metrics = self.model_fn(data)
        # backward
        if not self.enable_fgm:
            with torch.set_grad_enabled(True):
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
        # fgm adversarial training
        elif self.enable_fgm:
            with torch.set_grad_enabled(True):
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        self.scaler.scale(loss).backward()
                    self.attack()
                    _, loss_adv, _ = self.model_fn(data)
                    with torch.cuda.amp.autocast():
                        self.scaler.scale(loss_adv).backward()
                    self.restore()
                    with torch.cuda.amp.autocast():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()        # 反向传播，得到正常的grad
                    self.attack()          # 在embedding上添加对抗扰动
                    _, loss_adv, _ = self.model_fn(data)
                    loss_adv.backward()    # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    self.restore()         # 恢复embedding参数
                    self.optimizer.step()  # 梯度下降，更新参数

        # schedule
        if self.scheduler:
            if self.step_scheduler_after == "batch":
                self.scheduler.step()

        return output, loss, metrics

    def model_fn(self, data):
        "前向计算"
        for key, value in data.items():
            data[key] = value.to(self.device)
        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self(**data)
        else:
            output, loss, metrics = self(**data)
        return output, loss, metrics

    def validate_one_epoch(self, data_loader):
        self.eval()
        self.model_state = ModelState.VALID
        losses = MetricMeter()

        tqdm_loader = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tqdm_loader):
            # no backward
            self.train_state = TrainingState.VALID_STEP_START
            with torch.no_grad():
                _, self.step_loss, metrics = self.validate_one_step(data)
            self.train_state = TrainingState.VALID_STEP_END
            losses.update(self.step_loss.item())

            if b_idx == 0:
                metrics_meter = {k: MetricMeter() for k in metrics}

            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m])
                monitor[m_m] = metrics_meter[m_m].avg
            tqdm_loader.set_postfix(loss=losses.avg, stage="valid", **monitor)
            self.valid_step += 1

        tqdm_loader.close()

        self.metrics[self._model_state].update(monitor)
        self.metrics[self._model_state]["loss"] = losses.avg

        return losses.avg

    def validate_one_step(self, data):
        # forward only
        output, loss, metrics = self.model_fn(data)
        return output, loss, metrics

    def predict(self,
                dataset,
                batch_size=16,
                n_jobs=1):
        """预测函数。直接写一个处理单个输入的 predict 方法比这个简单，需要时可以重写

        Args:
            dataset (Iterable): 可产生预测输入可迭代对象，格式和训练阶段保持一致，符合forward方法的输入参数
            batch_size (int, optional): batch size. Defaults to 16.
            n_jobs (int, optional): 处理数据loader时，并行处理任务数. Defaults to 1.

        Yields:
            numpy.ndarray: 生成器方式输出结果
        """
        self.eval()
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  num_workers=n_jobs,
                                                  pin_memory=True)

        tqdm_loader = tqdm(data_loader, total=len(data_loader))
        for data in tqdm_loader:
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
                yield out

            tqdm_loader.set_postfix(stage="test")
        tqdm_loader.close()

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def attack(self):
        "用于NLP任务，对embedding层参数进行对抗训练，使用FGM算法"
        for name, param in self.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        "用于NLP任务，对embedding层参数进行对抗训练，对抗训练参数更新后，恢复原embedding"
        for name, param in self.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def save(self, model_path):
        "保存模型、优化器、lr调度器、训练基本状态"
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.epoch
        model_dict["fp16"] = self.fp16
        model_dict["metrics"] = self.metrics
        model_dict["best_scores"] = self.best_scores
        model_dict["the_one"] = self.the_one
        torch.save(model_dict, model_path)
        logger.info(f" [Save model at] {model_path}...")

    def load(self, model_path, device="cuda", strict=False):
        "恢复模型基本状态"
        self.device = device
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(model_dict["state_dict"], strict=False)
        self.optimizer.load_state_dict(model_dict["optimizer"])
        self.scheduler.load_state_dict(model_dict["scheduler"])
        self.epoch = model_dict["epoch"]
        self.fp16 = model_dict["fp16"]
        self.metrics = model_dict["metrics"]
        self.best_scores = model_dict["best_scores"]
        self.the_one = model_dict["the_one"]
        logger.info(f" [Restor model from] {model_path}...")

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        # 可以在设置model_state时，执行一些动作
        self._model_state = value
        logger.info(f" [In State] ### {value} EPOCH {self.epoch} ### ...")

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        """每次设置 train_state 值时，根据输入调用Callback对象中同名的 method，

        value取值：
            TrainingState.TRAIN_START -- 实际值为 字符串："on_train_start"
            TrainingState.TRAIN_END -- 实际值为 字符串："on_train_end"

            TrainingState.EPOCH_START -- 实际值为 字符串："on_epoch_start"
            TrainingState.EPOCH_END -- 实际值为 字符串："on_epoch_end"

            TrainingState.TRAIN_EPOCH_START -- 实际值为 字符串："on_train_epoch_start"
            TrainingState.TRAIN_EPOCH_END -- 实际值为 字符串："on_train_epoch_end"

            TrainingState.VALID_EPOCH_START -- 实际值为 字符串："on_valid_epoch_start"
            TrainingState.VALID_EPOCH_END -- 实际值为 字符串："on_valid_epoch_end"

            TrainingState.TRAIN_STEP_START -- 实际值为 字符串："on_train_step_start"
            TrainingState.TRAIN_STEP_END -- 实际值为 字符串："on_train_step_end"

            TrainingState.VALID_STEP_START -- 实际值为 字符串："on_valid_step_start"
            TrainingState.VALID_STEP_END -- 实际值为 字符串："on_valid_step_end"

            TrainingState.TEST_STEP_START -- 实际值为 字符串："on_test_step_start"
            TrainingState.TEST_STEP_END -- 实际值为 字符串："on_test_step_end"
        """
        self._train_state = value
        if self.callback_runner is not None:
            self.callback_runner(value)

    ######### 以下函数需要在使用时自定义 #########
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def monitor_metrics(self, *args, **kwargs):
        "计算评价指标的方法"
        raise NotImplementedError(
            "monitor_metrics method is not implemented !")

    def get_loss(self, *args, **kwargs):
        "定义损失函数计算的方法"
        raise NotImplementedError("get_loss method is not implemented !")

    def custom_optimizer(self, *args, **kwargs):
        "定义优化器"
        raise NotImplementedError(
            "custom_optimizer method is not implemented !")

    def custom_scheduler(self, *args, **kwargs):
        "定义学习率调节方法"
        logger.info("[Config] custom scheduler is not used")
        return None
