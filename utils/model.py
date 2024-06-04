# @Author   : ChaoQiezi
# @Time     : 2024/4/26  11:27
# @FileName : model.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放相关模型等
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from Config import DEVICE


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    def __init__(self, input_size=15, hidden_size=512, output_size=1, num_layers=3, dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):  # x.shape=(batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out.shape=(batch_szie, seq_len<123>, hidden_size<128>)
        """
        lstm_out: shape=(batch_size, seq_len, hidden_size)  # 最后一层所有时间步的输出
        h_n, c_n: shape=(num_layer, batch_size, hidden_size), 最后一个时间步的隐藏状态和细胞状态
        """

        reg_out = self.regression(h_n[-1, :, :])  # .squeeze(-1)  # 去除最后一个维度

        return reg_out


"""
由于增加多头注意力机制的LSTM是第一次尝试,所以代码中解释相对较多.
参考: https://mp.weixin.qq.com/s/Vs96H2hQy60qdRicO5ZRbg
"""


# 定义多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 定义多头注意力层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        """
        注意力头数(num_heads)应该能均匀地分割LSTM隐藏层的维度,也就是hidden_size能被
        num_heads整除
        """
        self.dropout = nn.Dropout(p=0.1)  # 随机失活层,避免attention时产生过拟合

    def forward(self, lstm_out):
        """
        MultiheadAttention期望输入的形状是batch_first=False即 shape=(seq_len, batch_size, hidden_size)
        而我们输入的lstm_output(也就是lstm层的输出)由于设置了batch_size=True因此 shape=(batch_size, seq_len, hidden_size)
        所以我们输入的lstm_output与要求的输入其shape不一致,需要转置维度
        """

        lstm_out = lstm_out.permute(1, 0, 2)  # 转置维度
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)  # 自注意力机制
        attn_output = self.dropout(attn_output)  # 应用随机失活层--dropout
        attn_output = attn_output.permute(1, 0, 2)  # 转置回原来的维度
        return attn_output, attn_weights


# 定义 Attention_LSTM 模型
class Attention_LSTM(nn.Module):
    def __init__(self, lstm_params, attention_params):
        super(Attention_LSTM, self).__init__()
        self.hidden_size = lstm_params['hidden_size']
        self.num_layers = lstm_params['num_layers']
        # 定义LSTM层
        self.lstm = nn.LSTM(lstm_params['input_size'], lstm_params['hidden_size'], lstm_params['num_layers'],
                            batch_first=True)
        # 定义多头注意力层
        self.attention = MultiHeadAttention(lstm_params['hidden_size'], attention_params['num_heads'])
        # 定义全连接层
        self.fc1 = nn.Linear(lstm_params['hidden_size'], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, lstm_params['output_size'])
        self.relu = nn.ReLU()  # 激活函数ReLU

    def forward(self, x):  # x.shape=(batch_size, seq_len, feature_size)
        # # 初始化隐藏状态和细胞状态
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        # # LSTM前向传播
        # lstm_out, _ = self.lstm(x, (h0, c0))

        # 应用LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)  # 不传入h0,c0默认即0填充(和上面没有本质区别)
        """
        lstm_out为最后一层LSTM层的所有时间步的输出,其shape=(batch_size, seq_len, hidden_size)
        h_n, c_n表示所有LSTM层的最后一个时间步的输出, 其shape=(num_layers, batch_size, hidden_size)
        """

        # 应用多头注意力层
        attn_out, _ = self.attention(lstm_out)  # 输入输出shape不变,但是经过注意力加权
        out = self.relu(attn_out[:, -1, :])  # 取多头注意力层输出的最后一个时间步

        # 全连接层前向传播
        out = self.relu(self.fc1(out))  # 全连接层1
        out = self.relu(self.fc2(out))  # 全连接层2
        out = self.relu(self.fc3(out))  # 全连接层3
        out = self.relu(self.fc4(out))  # 全连接层4
        out = self.fc5(out)  # 输出层

        return out


def train_epoch(model, data_loader, optimizer, loss_func=nn.MSELoss(), pbar=None):
    model.train()  # 训练模式

    # 每次迭代batch_size样本
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        # 清理存在NAN的样本
        valid_mask = ~(torch.isnan(inputs).any(dim=1).any(dim=1) | torch.isnan(targets).any(1))
        if valid_mask.any():  # 但凡有一个样本存在有效值
            inputs, targets = inputs[valid_mask], targets[valid_mask]
        else:
            continue

        optimizer.zero_grad()  # 清除存储梯度
        outputs = model(inputs)  # 模型预测
        loss = loss_func(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播,计算梯度
        optimizer.step()  # 更新权重

        if pbar is not None:
            pbar.set_postfix_str("Loss: {:.4f}".format(loss.item()))
    else:
        return loss.item()

# def train_epoch_ignore_nan(model, data_loader, optimizer, loss_func=nn.MSELoss(), pbar=None):
#     model.train()  # 训练模式
#
#     # 每次迭代batch_size样本
#     for inputs, targets in data_loader:
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#         # 检测nan
#         valid_mask = ~(torch.isnan(inputs).any(dim=1).any(dim=1) | torch.isnan(targets).any(1))
#         if not valid_mask.any():  # 所有样本均存在无效值
#             continue
#
#         optimizer.zero_grad()  # 清除存储梯度
#         outputs = model(inputs[[0]])  # 模型预测
#         loss = loss_func(outputs[valid_mask], targets[valid_mask])  # 计算损失,忽略存在nan的样本
#         loss.backward()  # 反向传播,计算梯度
#         optimizer.step()  # 更新权重
#
#         if pbar is not None:
#             pbar.set_postfix_str("Loss: {:.4f}".format(loss.item()))
#     else:
#         return loss.item()
