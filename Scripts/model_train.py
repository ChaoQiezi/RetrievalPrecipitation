# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:12
# @FileName : model_train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 准备
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model14_train_test.h5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 读取
with h5py.File(samples_path, 'r') as f:
    train_x, train_y, test_x, test_y = f['train_x'][:], f['train_y'][:], f['test_x'][:], f['test_y'][:]
train_x, train_y, test_x, test_y = torch.tensor(train_x, dtype=torch.float32), \
    torch.tensor(train_y, dtype=torch.float32), torch.tensor(test_x, dtype=torch.float32), \
    torch.tensor(test_y, dtype=torch.float32)
# DataLoader
train_ds = TensorDataset(train_x, train_y)
train_data_loader = DataLoader(train_ds, batch_size=16, shuffle=True)


class LSTMModelSame(nn.Module):
    def __init__(self, input_size=15, hidden_size=512, num_layers=3, output_size=1):
        super().__init__()
        # self.causal_conv1d = nn.Conv1d(input_size, 128, 5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):  # x.shape=(batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out.shape=(batch_szie, seq_len<123>, hidden_size<128>)
        reg_out = self.regression(lstm_out).squeeze(-1)  # .squeeze(-1)  # 去除最后一个维度

        return reg_out


import torch
import torch.nn as nn


class LSTMModelFuture(nn.Module):
    def __init__(self, input_size=15, hidden_size=512, output_size=1, num_layers=3, dropout_rate=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):  # x.shape=(batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out.shape=(batch_szie, seq_len<123>, hidden_size<128>)
        reg_out = self.regression(lstm_out[:, -1, :])  # .squeeze(-1)  # 去除最后一个维度

        return reg_out


# 创建模型
model = LSTMModelFuture().to(device)
# model = LSTMModelSame().to(device)
summary(model, input_data=(30, 15))  # 要求时间长度为14, 特征数为15, 输出模型结构
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 100
loss_epoch = []
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    loss_epoch.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
import matplotlib.pyplot as plt
plt.plot(loss_epoch)
plt.show()
# 使用训练好的模型进行预测
# DataLoader
test_ds = TensorDataset(test_x, test_y)
test_data_loader = DataLoader(test_ds, batch_size=16)
predictions, real_labels = [], []
model.eval()
with torch.no_grad():
    for input, labels in test_data_loader:
        predicted = model(input.to(device))

        predictions.append(predicted.detach().cpu().numpy())
        real_labels.append(labels.detach().cpu().numpy())

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
predictions = np.concatenate(predictions, axis=0)
real_labels = np.concatenate(real_labels, axis=0)

# mse_list = []
# rmse_list = []
# mae_list = []
# r2_list = []
# reals = []
# preds = []
# for row in range(predictions.shape[0]):
#     pred = predictions[row, :]
#     real = real_labels[row, :]
#     mse = mean_squared_error(real, pred)
#     # rmse = mean_squared_log_error(real, pred)
#     mae = mean_absolute_error(real, pred)
#     r2 = r2_score(real, pred)
#
#     mse_list.append(mse)
#     # rmse_list.append(rmse)
#     mae_list.append(mae)
#     r2_list.append(r2)
#     reals.append(real[0])
#     preds.append(pred[0])
# print('mse', np.mean(mse_list))
# print('rmse', np.mean(rmse_list))
# print('mae', np.mean(mae_list))
# print('r2', np.mean(r2_list))

reals = []
preds = []
for row in range(predictions.shape[0]):
    pred = predictions[row, :]
    real = real_labels[row, :]

    reals.append(real[0])
    preds.append(pred[0])
    # preds.append(pred[0] if pred[0] > 0 else 0)
print('mse', mean_squared_error(reals, preds))
# print('rmse', mean_squared_log_error(reals, preds))
print('mae', mean_absolute_error(reals, preds))
print('r2', r2_score(reals, preds))
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
ax1 = axs[0]
ax2 = axs[1]
ax1.plot(reals)
ax2.plot(preds)
plt.show()
# print(f'Predicted precipitation: {predicted}')

