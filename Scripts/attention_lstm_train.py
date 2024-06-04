# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:12
# @FileName : lstm_train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于基于注意力机制的lstm模型的训练
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

from utils.model import LSTMModelFuture, DEVICE, train_epoch, Attention_LSTM
import Config

# 准备
model_name = 'model1'
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\{}_train_test.h5'.format(model_name)
save_model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\{}_attn_lstm.pth'.format(model_name)

# 读取样本
with h5py.File(samples_path, 'r') as f:
    train_x, train_y = torch.tensor(f['train_x'][:]), torch.tensor(f['train_y'][:]),
    train_size, seq_len, feature_size = train_x.shape
    pred_len = train_y.shape[1]
# 数据加载器
train_ds = TensorDataset(train_x, train_y)  # Each sample will be retrieved by indexing tensors along the first dimension.
train_data_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
# 输出基本信息
print('当前训练特征项shape: {}\n当前训练目标项shape: {}'.format(train_x.shape, train_y.shape))
print('训练样本数目: {}\n单个样本特征项数: {}'.format(train_size, feature_size))
print('记忆期: {}; 预见期: {}'.format(seq_len, pred_len))

# 创建模型
model_params = {
    'lstm': {
        'input_size': feature_size,  # 输入特征维度
        'hidden_size': 256,              # LSTM隐藏层维度
        'num_layers': 2,                 # LSTM层数
        'output_size': 1                 # 输出维度
    },
    'attention': {
        'num_heads': 8                   # 注意力头数
    }
}
model = Attention_LSTM(model_params['lstm'], model_params['attention'])
summary(model, input_data=(seq_len, feature_size))  # 要求时间长度为14, 特征数为15, 输出模型结构
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

# 训练模型
loss_epoch = []
pbar = tqdm.tqdm(range(Config.num_epochs))
for epoch in pbar:
    pbar.set_description('Epoch: {:03}'.format(epoch))
    loss_v = train_epoch(model, train_data_loader, optimizer, loss_func=criterion, pbar=pbar)
    loss_epoch.append(loss_v)
torch.save(model.state_dict(), save_model_path)  # 保存模型
# 输出模型训练情况
fig, ax = plt.subplots(figsize=(11, 7))
plt.plot(loss_epoch)
ax.set_xlabel('Epoch 次数', fontsize=24)
ax.set_ylabel('MSE Loss', fontsize=24)
ax.set_title(f'LSTM({model_name}) training loss diagram', fontsize=30)
ax.legend(['MSE Loss'], fontsize=18)
ax.tick_params(labelsize=16)
ax.grid(linestyle='--', alpha=0.6)
plt.show()

