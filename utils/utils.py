# @Author   : ChaoQiezi
# @Time     : 2024/4/25  19:21
# @FileName : utils.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放常用工具更函数
"""

import numpy as np


def create_xy_same(dataset, window_size=14, step_size=1):
    """
    为时间序列创建滑动窗口生成样本, XY同时期
    :param dataset:待划分的数据集
    :param window_size: 时间滑动窗口大小
    :param step_size: 窗口移动步长
    :return: 元组(train_x, train_y, train_ix)
    """

    xs, ys, ixs = [], [], []
    for st_id in dataset['st'].unique():
        cur_data = dataset[dataset['st'] == st_id].reset_index(drop=True)
        for start in range(0, len(cur_data) - window_size + 1, step_size):
            end = start + window_size - 1
            xs.append(cur_data.loc[start:end, 'mwhs01':'mwhs15'])
            ys.append(cur_data.loc[start:end, 'PRCP'])
            ixs.append(cur_data.loc[start:end, ['st', 'ymdh']].apply(lambda x: str(x['st']) + '_' + x['ymdh'].strftime('%Y%m%d'), axis=1))
    train_x = np.array(xs)
    train_y = np.array(ys)
    train_ix = np.array(ixs)

    return train_x, train_y, train_ix


def create_xy_future(dataset, window_size=14, step_size=1, future_size=1):
    """
    为时间序列基于滑动窗口生成样本, X和Y不同时期
    :param dataset: 待划分的数据
    :param window_size: 时间窗口的大小(理解为LSTM的记忆时间)
    :param step_size: 窗口移动的步长(避免样本之间的高度相似性)
    :param future_size: y对应的未来大小(对应LSTM的可预见期)
    :return: 元组(train_x, train_y)
    """

    xs, ys = [], []
    for st_id in dataset['st'].unique():
        cur_data = dataset[dataset['st'] == st_id].reset_index(drop=True)
        for x_start in range(0, len(cur_data) - (window_size + future_size) + 1, step_size):
            x_end = x_start + window_size - 1  # -1是因为.loc两边为闭区间
            y_start = x_end
            y_end = y_start + future_size - 1
            x_col = ['PRCP'] + ['mwhs{:02d}'.format(_ix) for _ix in range(1, 16)]
            xs.append(cur_data.loc[x_start:x_end, x_col])
            ys.append(cur_data.loc[y_start:y_end, 'PRCP'])
    train_x = np.array(xs)
    train_y = np.array(ys)

    return train_x, train_y

