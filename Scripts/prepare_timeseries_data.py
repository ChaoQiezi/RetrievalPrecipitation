# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:00
# @FileName : prepare_timeseries_data.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 制作可供训练输入的时间序列样本
"""


import h5py
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from utils.utils import create_xy_same, create_xy_future

# 准备
in_path = r'H:\Datasets\Objects\retrieval_prcp\Data\PRCP_cq34_FY3DL2_202005_08_daily_T_order.csv'
out_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model1_train_test.h5'

# 读取
df = pd.read_csv(in_path)
df['ymdh'] = pd.to_datetime(df['ymdh'], format='%Y%m%d%H')  # 转换成时间对象
# 数据集划分和标准化
split_time = datetime(2020, 7, 10)  # 划分时间节点, 5~7月为训练集, 8月为验证集, 约为3:1
train_ds = df[df['ymdh'] <= split_time]
test_ds = df[df['ymdh'] > split_time]
scaler = MinMaxScaler()  # 标准化器
train_ds.loc[:, 'mwhs01':'mwhs15'] = scaler.fit_transform(train_ds.loc[:, 'mwhs01':'mwhs15'])
test_ds.loc[:, 'mwhs01':'mwhs15'] = scaler.transform(test_ds.loc[:, 'mwhs01':'mwhs15'])  # 注意标准化不能独立对测试集进行, 标准化参数应来源于训练集
train_x, train_y = create_xy_future(train_ds, window_size=30, step_size=1, future_size=7)
test_x, test_y = create_xy_future(test_ds, window_size=30, step_size=1, future_size=7)
# train_x, train_y, _ = create_xy_same(train_ds, window_size=14, step_size=1)
# test_x, test_y, _ = create_xy_same(test_ds, window_size=14, step_size=1)
# 输出为HDF5文件
with h5py.File(out_path, 'w') as f:
    f.create_dataset('train_x', data=train_x)
    f.create_dataset('train_y', data=train_y)
    f.create_dataset('test_x', data=test_x)
    f.create_dataset('test_y', data=test_y)




