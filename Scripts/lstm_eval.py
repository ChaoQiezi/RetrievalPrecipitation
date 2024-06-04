# @Author   : ChaoQiezi
# @Time     : 2024/5/15  19:28
# @FileName : lstm_eval.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于模型评估/预测
"""


from utils.model import LSTMModelFuture
from utils.utils import decode_time_col, plot_comparison, cal_nse
import Config

import os
import tqdm
import h5py
import torch
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader


# 准备
model_name = 'model19'
samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\{}_train_test.h5'.format(model_name)
model_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\ModelStorage\{}_V01.pth'.format(model_name)

# 读取测试样本
with h5py.File(samples_path) as f:
    test_x, test_y, test_ix = torch.tensor(f['test_x'][:]), torch.tensor(f['test_y'][:]), f['test_ix'][:]
    test_size, seq_len, feature_size = test_x.shape
test_ix = decode_time_col(test_ix)

# 加载模型
model = LSTMModelFuture(feature_size, output_size=Config.pred_len_day).to(Config.DEVICE)
model.load_state_dict(torch.load(model_path))  # 加载存储模型
model.eval()  # 评估模式
with torch.no_grad():
    for station_name in test_ix['站名'].unique():
        # 预测
        temp_ix = test_ix[test_ix['站名'] == station_name][[x for x in test_ix.columns if x != '站名']]
        temp_x = test_x[test_ix['站名'] == station_name].to(Config.DEVICE)
        temp_y_obs = test_y[test_ix['站名'] == station_name]
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
        save_path = os.path.join(r'I:\PyProJect\RetrievalPrecipitation\Assets\Charts',
                                 'pred_obs_train_{}_m{}day_p{}day.png'.format(station_name, Config.seq_len_day,
                                                                             Config.pred_len_day))
        plot_comparison(temp_ix, temp_y_obs, temp_y_pred, station_name, save_path=save_path)

