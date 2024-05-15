# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:12
# @FileName : model_train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于模型的训练
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils.model import LSTMModelFuture, LSTMModelSame, DEVICE, train_epoch
import Config

# 准备
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model14_train_test.h5'
save_model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\LSTM_V01.pth'
num_epochs = Config.num_epochs  # 训练次数
lr = Config.lr  # 学习率
batch_size = Config.batch_size  # 批次大小
loss_epoch = []

# 读取样本
with h5py.File(samples_path, 'r') as f:
    train_x, train_y, test_x, test_y = \
        torch.tensor(f['train_x']), torch.tensor(f['train_y']), torch.tensor(f['test_x']), torch.tensor(f['test_y'])
    train_size, seq_len, feature_size = train_x.shape
# 数据加载器
train_ds = TensorDataset(train_x, train_y)  # Each sample will be retrieved by indexing tensors along the first dimension.
train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# 输出基本信息
print('当前训练特征项shape: {}\n当前训练目标项shape: {}'.format(train_x.shape, train_y.shape))
print('训练样本数目: {}\n单个样本时间长度: {}\n单个样本特征项数: {}'.format(train_size, seq_len, feature_size))
print('预测期数: {}'.format(train_y.shape[1]))

# 创建模型
model = LSTMModelFuture().to(DEVICE)
# model = LSTMModelSame().to(device)
summary(model, input_data=(seq_len, feature_size))  # 要求时间长度为14, 特征数为15, 输出模型结构
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型
pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:
    pbar.set_description('Epoch: {:03}'.format(epoch))
    loss_v = train_epoch(model, train_data_loader, optimizer, loss_func=criterion)
    pbar.set_postfix_str("Loss: {:.4f}".format(loss_v))
    loss_epoch.append(loss_v)
torch.save(model.state_dict(), save_model_path)  # 保存模型
# 输出模型训练情况
fig, ax = plt.subplots(figsize=(11, 7))
plt.plot(loss_epoch)
ax.set_xlabel('Epoch 次数', fontsize=24)
ax.set_ylabel('MSE Loss', fontsize=24)
ax.set_title('LSTM training loss diagram', fontsize=30)
ax.legend(['MSE Loss'], fontsize=18)
ax.tick_params(labelsize=16)
ax.grid(linestyle='--', alpha=0.6)
plt.show()

