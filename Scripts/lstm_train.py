# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:12
# @FileName : lstm_train.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于模型的训练
"""

from utils.model import LSTMModelFuture, DEVICE, train_epoch
from utils.utils import decode_time_col, plot_comparison, cal_nse
import Config

import os
import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader


# 准备
model_name = 'model4'
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\{}_train_test.h5'.format(model_name)
save_model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\{}_V01.pth'.format(model_name)

# 读取样本
with h5py.File(samples_path, 'r') as f:
    train_x, train_y, train_ix = torch.tensor(f['train_x'][:]), torch.tensor(f['train_y'][:]), f['train_ix'][:]
    train_size, seq_len, feature_size = train_x.shape
train_ix = decode_time_col(train_ix)
# 数据加载器
train_ds = TensorDataset(train_x, train_y)  # Each sample will be retrieved by indexing tensors along the first dimension.
train_data_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
# 输出基本信息
print('当前训练特征项shape: {}\n当前训练目标项shape: {}'.format(train_x.shape, train_y.shape))
print('训练样本数目: {}\n单个样本时间长度: {}\n单个样本特征项数: {}'.format(train_size, seq_len, feature_size))
print('预测期数: {}'.format(train_y.shape[1]))

# 创建模型
model = LSTMModelFuture(feature_size, output_size=Config.pred_len_day).to(DEVICE)
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
# 绘制拟合情况和评估
model.load_state_dict(torch.load(save_model_path))  # 加载存储模型
model.eval()  # 评估模式
with torch.no_grad():
    for station_name in train_ix['站名'].unique():
        # 预测
        temp_ix = train_ix[train_ix['站名'] == station_name][[x for x in train_ix.columns if x != '站名']]
        temp_x = train_x[train_ix['站名'] == station_name].to(Config.DEVICE)
        temp_y_obs = train_y[train_ix['站名'] == station_name]
        temp_y_pred = model(temp_x).detach().cpu().numpy()
        temp_y_pred[temp_y_pred < 0] = 0  # 负数替换为0
        # 反归一化
        scaler = joblib.load(Config.scalers_path)['{}_y_scaler'.format(model_name)]
        # temp_y_obs = scalers['model__y_scaler'].inverse_transform(pd.DataFrame({Config.target_name[0]: temp_y_obs})).squeeze()
        # temp_y_pred = scalers['model__y_scaler'].inverse_transform(pd.DataFrame({Config.target_name[0]: temp_y_pred})).squeeze()
        temp_y_obs = scaler.inverse_transform(temp_y_obs)
        temp_y_pred = scaler.inverse_transform(temp_y_pred)
        # 计算训练集的评估指标
        r2 = r2_score(temp_y_obs, temp_y_pred)
        rmse = mean_squared_log_error(temp_y_obs, temp_y_pred)
        nse = cal_nse(temp_y_obs, temp_y_pred)
        print('训练集评估结果--站名: {}; R2: {:0.4f}; RMSE: {:0.4f}; NSE: {:0.4f}'.format(station_name, r2, rmse, nse))
        # 绘制
        # 合并重叠部分(简单均值)
        temp_y_pred = pd.DataFrame(temp_y_pred)
        for ix in range(temp_y_pred.shape[1]):
            temp_y_pred[ix] = temp_y_pred[ix].shift(ix)
        temp_y_pred = np.nanmean(temp_y_pred, axis=1)
        temp_y_obs = temp_y_obs[:, 0]  # 直接取第一列即可
        temp_ix = temp_ix['0_date']
        # combined_preds = np.zeros(temp_y_pred.shape[0] + Config.pred_len_day - 1)
        # counts = np.zeros_like(combined_preds)
        # for ix, line in enumerate(temp_y_pred):
        #     combined_preds[ix:ix+Config.pred_len_day] += line
        #     counts[ix:ix+Config.pred_len_day] += 1
        # combined_preds /= counts
        #
        # combined_obss = np.zeros(temp_y_obs.shape[0] + Config.pred_len_day - 1)
        # counts = np.zeros_like(combined_obss)
        # for ix, line in enumerate(temp_y_obs):
        #     combined_obss[ix:ix + Config.pred_len_day] += line
        #     counts[ix:ix + Config.pred_len_day] += 1
        # combined_obss /= counts

        # save_path = os.path.join(Config.Assets_charts_dir, 'pred_obs_train_{}.png'.format(station_name))
        save_path = os.path.join(r'I:\PyProJect\RetrievalPrecipitation\Assets',
                                 'pred_obs_train_{}_m{}day_p{}day.png'.format(station_name, Config.seq_len_day,
                                                                             Config.pred_len_day))
        plot_comparison(temp_ix, temp_y_obs, temp_y_pred, station_name, save_path=save_path)


