# @Author   : ChaoQiezi
# @Time     : 2024/5/15  19:28
# @FileName : lstm_eval.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于基于自注意力机制的lstm模型的预测和评估
"""

import tqdm
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from utils.model import LSTMModelFuture, Attention_LSTM
from utils.utils import plot_comparison, decode_time_col
import Config
# 准备
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model1_train_test.h5'

model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\model1_attn_lstm.pth'

# 读取测试样本
with h5py.File(samples_path) as f:
    test_x, test_y, test_ix = torch.tensor(f['test_x'][:]), torch.tensor(f['test_y'][:]), f['test_ix'][:]
test_ds = TensorDataset(test_x, test_y)
test_data_loader = DataLoader(test_ds, batch_size=16)
test_ix = decode_time_col(test_ix)

# 加载模型
# 创建模型
model_params = {
    'lstm': {
        'input_size': test_x.shape[2],  # 输入特征维度
        'hidden_size': 256,              # LSTM隐藏层维度
        'num_layers': 2,                 # LSTM层数
        'output_size': 1                 # 输出维度
    },
    'attention': {
        'num_heads': 8                   # 注意力头数
    }
}
model = Attention_LSTM(model_params['lstm'], model_params['attention']).to(Config.DEVICE)
model.load_state_dict(torch.load(model_path))

# 评估
preds, reals = [], []
model.eval()  # 设置为评估模式
with torch.no_grad():
    pbar = tqdm.tqdm(test_data_loader)  # 进度条
    pbar.set_description('The model is being evaluated')
    for input, real in pbar:
        # 预测
        pred = model(input.to(Config.DEVICE))

        preds.append(pred.detach().cpu())
        reals.append(real.detach().cpu())

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error

preds = np.concatenate(preds, axis=0)
reals = np.concatenate(reals, axis=0)
valid_mask = ~np.isnan(preds)
preds = preds[valid_mask]
reals = reals[valid_mask]

print('mse', mean_squared_error(reals, preds))
print('mae', mean_absolute_error(reals, preds))
print('r2', r2_score(reals, preds))
# fig, axs = plt.subplots(2, 1)
# axs = axs.flatten()
# ax1 = axs[0]
# ax2 = axs[1]
# ax1.plot(reals)
# ax2.plot(preds)

# ax1.set_xlabel('Date')
# ax1.set_ylabel('Real')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Pred')
# plt.show()
for station_name in test_ix['站名'].unique():
    cur_ix = test_ix['站名'] == station_name
    plot_comparison(test_ix['0_date'][cur_ix], reals[cur_ix], preds[cur_ix], station_name)
    break
