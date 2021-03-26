# JEM: https://arxiv.org/abs/1912.03263
# Git: https://github.com/tohmae/pytorch-jem
# 和 adversarial train method 是类似的，只是是从优化目标入手，希望模型预测输出的energy尽量小，真实图片energy尽量大。
# 问题在于，它比较慢，有时不稳定 nan

import random
import numpy as np
import torch


class ReplayBuffer:
    "jem中缓存历史结果的类，从中采样，优化energy相关损失，希望模型预测输出的energy尽量小，真实图片energy尽量大"

    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}
        self.init_length = 0

    def __len__(self):
        return self.cur_size

    def seed_buffer(self, episodes):
        self.init_length = len(episodes)
        self.add(episodes, np.ones(self.init_length))

    def add(self, episodes, *args):
        """Add episodes to buffer."""
        idx = 0
        while self.cur_size < self.max_size and idx < len(episodes):
            self.buffer[self.cur_size] = episodes[idx]
            self.cur_size += 1
            idx += 1

        if idx < len(episodes):
            remove_idxs = self.remove_n(len(episodes) - idx)
            for remove_idx in remove_idxs:
                self.buffer[remove_idx] = episodes[idx]
                idx += 1

        assert len(self.buffer) == self.cur_size

    def remove_n(self, n):
        """Get n items for removal."""
        # random removal
        idxs = random.sample(range(self.init_length, self.cur_size), n)
        return idxs

    def get_batch(self, n):
        """Get batch of episodes to train on."""
        # random batch
        idxs = random.sample(range(self.cur_size), n)
        return [self.buffer[idx] for idx in idxs]

    def update_last_batch(self, delta):
        pass


def LogSumExp(x):
    x = torch.logsumexp(x, 1)  # energy base model中常见
    x = x.view(len(x), 1)
    return x


def Sample(func_of_x,
           batch_size,
           dim,
           buffer,
           device,
           rou=0.05,
           alpha=1.0,
           eta=20,
           sigma=0.01):
    m_uniform = torch.distributions.uniform.Uniform(torch.tensor([-1.0]),
                                                    torch.tensor([1.0]))
    m_normal = torch.distributions.normal.Normal(torch.tensor([0.0]),
                                                 torch.tensor([1.0]))

    batch_size1 = int(batch_size * (1 - rou))
    batch_size2 = batch_size - batch_size1

    # random and history for robustness
    x1 = torch.stack(buffer.get_batch(batch_size1))
    x2 = m_uniform.sample((batch_size2, dim)).squeeze()
    x = torch.cat([x1, x2], dim=0)
    x = x.to(device)

    #  Stochastic Gradient Langevin Dynamics
    x.requires_grad_(True)
    for i in range(eta):
        jac = jacobian(func_of_x, x)
        if torch.isnan(jac).any():
            print("jac nan")
            exit(1)
        x = x + alpha * jac + sigma * m_normal.sample(x.shape).squeeze().to(device)

    x = x.detach()
    buffer.add(x.cpu())
    return x


def jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.
    This is according to the reverse mode autodiff rule,
    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,
    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.
    :param f: function R^N -> R
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N]
    """
    B, N = x.shape
    y = f(x)
    v = torch.zeros_like(y)
    v[:, 0] = 1.

    dy_i_dx = torch.autograd.grad(y,
                                  x,
                                  grad_outputs=v,
                                  retain_graph=True,
                                  create_graph=True,
                                  allow_unused=True)[0]  # shape [B, N]
    return dy_i_dx


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
    """
    from ignite.utils import convert_tensor

    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_supervised_trainer2(model, optimizer, loss_fn,
                              replay_buffer,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch):
    "使用 ignite 包，不用 ignite 时，需要修改该函数"
    from ignite.engine.engine import Engine
    from ignite.metrics import Average

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()

        LogSumExpf = lambda x: LogSumExp(model(x))
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        x = x.detach()
        y_pred = model(x)
        loss_elf = loss_fn(y_pred, y)
        x_sample = Sample(LogSumExpf, x.shape[0], x.shape[1], replay_buffer, device)
        replay_buffer.add(x_sample.cpu())
        loss_gen = -(LogSumExpf(x) - LogSumExpf(x_sample)).mean()
        loss = loss_elf + loss_gen
        loss.backward()
        optimizer.step()
        return {'loss':loss.item(), 'loss_elf':loss_elf.item(), 'loss_gen':loss_gen.item()}

    engine = Engine(_update)
    metric_loss = Average(output_transform=lambda output: output['loss'])
    metric_loss_elf = Average(output_transform=lambda output: output['loss_elf'])
    metric_loss_gen = Average(output_transform=lambda output: output['loss_gen'])
    metric_loss.attach(engine, "loss")
    metric_loss_elf.attach(engine, "loss_elf")
    metric_loss_gen.attach(engine, "loss_gen")

    return engine