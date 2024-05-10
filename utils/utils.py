# @Author   : ChaoQiezi
# @Time     : 2024/4/25  19:21
# @FileName : utils.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放常用工具和函数
"""

import h5py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# 初始化参数
split_time = datetime(2020, 7, 10)  # 划分时间节点, 5~7月为训练集, 8月为验证集, 约为3:1
time_name = None


def create_xy_same(dataset, x_col_names, y_col_name, window_size=30, step_size=1, st_col_name='st'):
    """
    为时间序列创建滑动窗口生成样本, XY同时期
    :param dataset:待划分的数据集
    :param window_size: 时间滑动窗口大小
    :param step_size: 窗口移动步长
    :return: 元组(train_x, train_y, train_ix)
    """
    global time_name

    xs, ys, ixs = [], [], []
    for st_id in dataset[st_col_name].unique():
        cur_data = dataset[dataset[st_col_name] == st_id].reset_index(drop=True)
        for start in range(0, len(cur_data) - window_size + 1, step_size):
            end = start + window_size - 1
            xs.append(cur_data.loc[start:end, x_col_names])
            ys.append(cur_data.loc[start:end, y_col_name])
            ixs.append(cur_data.loc[start:end, [time_name]].apply(lambda x: x[time_name].strftime('%Y%m%d'), axis=1))
            # ixs.append(cur_data.loc[start:end, ['st', 'ymdh']].apply(
            #     lambda x: str(x['st']) + '_' + x['ymdh'].strftime('%Y%m%d'), axis=1))
    xs = np.array(xs)
    ys = np.array(ys)
    ixs = np.array(ixs)

    # return train_x, train_y, train_ix
    return xs, ys, ixs


def create_xy_future(dataset, x_col_names, y_col_name, window_size=30, step_size=1, future_size=1, st_col_name='st'):
    """
    为时间序列基于滑动窗口生成样本, X和Y不同时期
    :param dataset: 待划分的数据
    :param window_size: 时间窗口的大小(理解为LSTM的记忆时间)
    :param step_size: 窗口移动的步长(避免样本之间的高度相似性)
    :param future_size: y对应的未来大小(对应LSTM的可预见期)
    :return: 元组(train_x, train_y)
    """
    global time_name

    xs, ys, ixs = [], [], []
    for st_id in dataset[st_col_name].unique():  # 迭代站点
        cur_data = dataset[dataset[st_col_name] == st_id].reset_index(drop=True)
        for x_start in range(0, len(cur_data) - (window_size + future_size) + 1, step_size):
            x_end = x_start + window_size - 1  # -1是因为.loc两边为闭区间
            y_start = x_end + 1
            y_end = y_start + future_size - 1
            # x_cols = ['PRCP'] + ['mwhs{:02d}'.format(_ix) for _ix in range(1, 16)]
            xs.append(cur_data.loc[x_start:x_end, x_col_names])
            ys.append(cur_data.loc[y_start:y_end, y_col_name])
            # ixs.append(cur_data.loc[y_start:y_end, time_name])
            ixs.append(cur_data.loc[y_start:y_end, [time_name]].apply(lambda x: x[time_name].strftime('%Y%m%d'), axis=1))
    xs = np.array(xs)
    ys = np.array(ys)
    ixs = np.array(ixs)

    return xs, ys, ixs


# def generate_samples(df, x_col_names, y_col_name, out_path, time_col_name='ymdh', format_str='%Y%m%d%H',
#                      split_time=split_time, is_same_periods=False, window_size=30, step_size=1, future_size=1,
#                      st_col_name='st'):
def generate_samples(df, x_col_names, y_col_name, out_path, time_col_name='ymdh', format_str='%Y%m%d%H',
                     split_time=split_time, is_same_periods=False, **kwargs):
    global time_name
    time_name = time_col_name

    df[time_col_name] = pd.to_datetime(df[time_col_name], format=format_str)  # 转换成时间对象
    # 训练测试集划分
    train_ds = df[df[time_col_name] <= split_time]
    test_ds = df[df[time_col_name] > split_time]
    # 标准化
    scaler = MinMaxScaler()  # 标准化器
    train_ds.loc[:, x_col_names] = scaler.fit_transform(train_ds.loc[:, x_col_names])
    test_ds.loc[:, x_col_names] = scaler.transform(test_ds.loc[:, x_col_names])  # 注意标准化不能独立对测试集进行, 标准化参数应来源于训练集
    # 特征项(x/features)和目标项(y/targets)划分
    if not is_same_periods:
        train_x, train_y, train_ix = create_xy_future(train_ds, x_col_names, y_col_name, **kwargs)
        test_x, test_y, test_ix = create_xy_future(test_ds, x_col_names, y_col_name, **kwargs)
    else:
        train_x, train_y, train_ix = create_xy_same(train_ds, x_col_names, y_col_name, **kwargs)
        test_x, test_y, test_ix = create_xy_same(test_ds, x_col_names, y_col_name, **kwargs)

    # 输出为HDF5文件
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('train_x', data=train_x)
        f.create_dataset('train_y', data=train_y)
        f.create_dataset('train_ix', data=train_ix)
        f.create_dataset('test_x', data=test_x)
        f.create_dataset('test_y', data=test_y)
        f.create_dataset('test_ix', data=test_ix)
