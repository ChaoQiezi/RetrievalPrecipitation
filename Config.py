# @Author   : ChaoQiezi
# @Time     : 2024/5/15  21:16
# @FileName : Config.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import matplotlib.pyplot as plt
import torch
from datetime import datetime

# 设置相关
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 可正常显示中文
# plt.rcParams['font.family'] = 'Simhei'
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = True  # 显示正负号

# 初始化参数
split_time = datetime(2020, 7, 10)  # 数据集的划分时间节点, 5~7月为训练集, 8月为验证集, 约为3:1

# 模型相关
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 50  # 训练次数
lr = 1e-4  # 学习率
batch_size = 16  # 批次大小