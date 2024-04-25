# @Author   : ChaoQiezi
# @Time     : 2024/4/25  19:21
# @FileName : utils.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放常用工具更函数
"""

import numpy as np


def create_xy(dataset, window_size=14, step_size=1):
    """
    为时间序列创建滑动窗口生成样本
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