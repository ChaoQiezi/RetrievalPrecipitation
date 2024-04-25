# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:12
# @FileName : model_train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import h5py
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 准备
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model1_train_test.h5'
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


class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=256, num_layers=3, output_size=10):
        super().__init__()
        # self.causal_conv1d = nn.Conv1d(input_size, 128, 5)
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):  # x.shape=(batch_size, seq_len, input_size)
        conv1d_out = self.causal_conv1d(F.pad(torch.transpose(x, 1, 2), (2, 0)))
        lstm_out, (h_n, c_n) = self.lstm(torch.transpose(conv1d_out, 1, 2))  # lstm_out.shape=(batch_szie, seq_len<123>, hidden_size<128>)
        reg_out = self.regression(lstm_out).squeeze(-1)  # .squeeze(-1)  # 去除最后一个维度

        return reg_out


# 创建模型
model = LSTMModel().to(device)
summary(model, input_data=(10, 15))  # 要求时间长度为14, 特征数为15, 输出模型结构
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 100
loss_epoch = []
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
# 假设test_x是要预测的特征集
test_x = torch.tensor([[280.40, 216.15, 208.33]], dtype=torch.float32)
test_x = test_x.view(1, 1, -1)
predicted = model(test_x).item()

print(f'Predicted precipitation: {predicted}')
