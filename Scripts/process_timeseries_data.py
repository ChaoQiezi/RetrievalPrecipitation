# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:00
# @FileName : process_timeseries_data.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 制作可供训练输入的时间序列样本

Note:
    model_1: FY3D-MWHS 1-15, daily, 5-8month, not exist NAN
    model_4: FY4A cn1-14, hourly, 5-8month, exists NAN
    model_14: FY4A cn1-14, daily, 5-8month, not exist NAN
    model_19: FY3D-MWHS 1-15 + FY4A cn1-14, daily, 5-8month, not exist NAN
"""

import pandas as pd
from utils.utils import generate_samples

# 准备
fy3d_daily_path = r'H:\Datasets\Objects\retrieval_prcp\Data\PRCP_cq34_FY3DL2_202005_08_daily_T_order.csv'
fy4a_daily_path = r'H:\Datasets\Objects\retrieval_prcp\Data\PRCP_cq34_FY4AL1_202005_08_daily_T_order_noNA.csv'
fy4a_hourly_path = r'H:\Datasets\Objects\retrieval_prcp\Data\PRCP_cq34_FY4AL1_202005_08_hour_merge_T_order.csv'
out_model1_samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model1_train_test.h5'
out_model4_samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model4_train_test.h5'
out_model14_samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model14_train_test.h5'
out_model19_samples_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model19_train_test.h5'
# x_cols_name = ['PRCP'] + ['mwhs{:02d}'.format(_ix) for _ix in range(1, 16)]
y_col_name = 'PRCP'

# 读取
fy3d_df = pd.read_csv(fy3d_daily_path)
fy4a_daily_df = pd.read_csv(fy4a_daily_path)
fy4a_hourly_df = pd.read_csv(fy4a_hourly_path)
# 生成模型训练样本
# model 1
fy3d_cols_name = ['mwhs{:02d}'.format(_ix) for _ix in range(1, 16)]
generate_samples(fy3d_df, fy3d_cols_name, y_col_name, out_model1_samples_path)
# model 4
fy4a_cols_name = ['cn{:02d}'.format(_ix) for _ix in range(1, 15)]
generate_samples(fy4a_hourly_df, fy4a_cols_name, y_col_name, out_model4_samples_path)
# model 14
generate_samples(fy4a_daily_df, fy4a_cols_name, y_col_name, out_model14_samples_path)
# model 19
model19_cols_name = fy3d_cols_name + fy4a_cols_name
fy3d_df['st_time'] = fy3d_df[['st', 'ymdh']].apply(lambda x: str(x['st']) + '_' + str(x['ymdh']), axis=1)
fy4a_daily_df['st_time'] = fy4a_daily_df[['st', 'ymdh']].apply(lambda x: str(x['st']) + '_' + str(x['ymdh']), axis=1)
model19_df = pd.merge(fy3d_df, fy4a_daily_df, on='st_time', suffixes=("", "_copy"))
generate_samples(model19_df, model19_cols_name, y_col_name, out_model19_samples_path)





