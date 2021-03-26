# JEM: https://arxiv.org/abs/1912.03263
# 和 adversarial train method 是类似的，只是是从优化目标入手，希望模型预测输出的energy尽量小，真实图片energy尽量大。
# 运行较慢，有时不稳定 nan

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np


eta = 20
alpha = 1.0
sigma = 0.01
buffer_size = 1000
rou = 0.05

B = utils.ReplayBuffer(buffer_size)
m_uniform = torch.distributions.uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
B.add(m_uniform.sample((100, 784)).squeeze())


class Net(nn.Module):
    def __init__(self, n_units, n_out):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, n_units)
        self.l2 = nn.Linear(n_units, n_units)
        self.l3 = nn.Linear(n_units, n_out)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch):
    model_obj.train()

    running_loss = 0
    step = 0
    for data, targets in loader_train:
        if step % 100 == 0:
            print("step={}".format(step))
        step += 1

        LogSumExpf = lambda x: utils.LogSumExp(model_obj(x))

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model_obj(data)
        loss_elf = loss_fn(outputs, targets)

        ### MAIN Difference ###
        # energy constraints
        data_sample = utils.Sample(LogSumExpf, data.shape[0], data.shape[1], buffer=B, device)
        loss_gen = -(LogSumExpf(data) - LogSumExpf(data_sample)).mean()
        loss = loss_elf + loss_gen
        #######################

        if torch.isnan(loss_elf):
            print("loss_elf nan")
            exit(1)
        if torch.isnan(loss_gen):
            print("loss_gen nan")
            exit(1)

        running_loss += loss.item()
        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()

    train_loss = running_loss / len(loader_train)
    print ('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, train_loss))


def test(loader_test, trained_model, loss_fn, device):
    trained_model.eval()
    correct = 0
    running_loss = 0
    with torch.no_grad():
        for data, targets in loader_test:

            data = data.to(device)
            targets = targets.to(device)

            outputs = trained_model(data)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

            loss = loss_fn(outputs, targets)
            running_loss += loss.item()

    data_num = len(loader_test.dataset)
    val_loss = running_loss / len(loader_test)
    print('\nAccuracy: {}/{} ({:.1f}%) loss: {:.4f}\n'.format(correct, data_num, 100. * correct / data_num, val_loss))


def main():

    # 1. GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 2. 配置
    batch_size = 100
    num_classes = 10
    epochs = 20

    ##########################################################################################
    # 3. MNIST
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1,)

    # 4. [0, 1]
    x = mnist.data / 255
    y = mnist.target.astype(np.int32)

    ##########################################################################################
    # 5. DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)

    x_train = x_train.reshape(60000, 28 * 28)
    x_test = x_test.reshape(10000, 28 *28)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(x)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(x_train.max(), x_train.min())
    print(x_test.max(), x_test.min())

    x_train = torch.Tensor(x_train)
    # x_train.requires_grad_()
    x_test = torch.Tensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    ##########################################################################################

    # 6. model
    model = Net(1000, n_out=num_classes).to(device)
    print(model) # ネットワークの詳細を確認用に表示

    # 7. 损失
    loss_fn = nn.CrossEntropyLoss()

    # 8. 优化器
    from torch import optim
    optimizer = optim.Adam(model.parameters())

    # 9. 运行
    print('Begin train')
    for epoch in range(1, epochs+1):
        train(loader_train, model, optimizer, loss_fn, device, epochs, epoch)
        test(loader_test, model, loss_fn, device)



if __name__ == '__main__':
    main()