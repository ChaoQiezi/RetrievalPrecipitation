# @Author   : ChaoQiezi
# @Time     : 2024/4/24  21:00
# @FileName : process_timeseries.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 制作可供训练输入的时间序列样本

Note:
    model_1: FY3D-MWHS 1-15, daily, 5-8month, not exist NAN
    model_4: FY4A cn1-14, hourly, 5-8month, exists NAN
    model_14: FY4A cn1-14, daily, 5-8month, not exist NAN
    model_19: FY3D-MWHS 1-15 + FY4A cn1-14, daily, 5-8month, not exist NAN
"""

from utils.utils import generate_samples, fast_viewing
import Config


import pandas as pd


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
fy3d_df['date'] = pd.to_datetime(fy3d_df['ymdh'], format='%Y%m%d%H')
# 生成模型训练样本
# model 1(daily)
fy3d_cols_name = ['PRCP'] + ['mwhs{:02d}'.format(_ix) for _ix in range(1, 16)]
fast_viewing(fy3d_df, fy3d_df['st'].unique()[:3], fy3d_cols_name)  # 简单展示3个站点的各个特征项和目标项随时间变化
generate_samples(fy3d_df, fy3d_cols_name, y_col_name, out_model1_samples_path, model_fix=1)
# model 4(hourly, exist NAN)
fy4a_cols_name = ['PRCP'] + ['cn{:02d}'.format(_ix) for _ix in range(1, 15)]
generate_samples(fy4a_hourly_df, fy4a_cols_name, y_col_name, out_model4_samples_path, window_size=Config.seq_len_hour,
                 future_size=Config.pred_len_hour, model_fix=4)
# model 14(daily)
generate_samples(fy4a_daily_df, fy4a_cols_name, y_col_name, out_model14_samples_path, model_fix=14)
# model 19(daily)
model19_cols_name = fy3d_cols_name + fy4a_cols_name
model19_cols_name.remove('PRCP')
fy3d_df['st_time'] = fy3d_df[['st', 'ymdh']].apply(lambda x: str(x['st']) + '_' + str(x['ymdh']), axis=1)
fy4a_daily_df['st_time'] = fy4a_daily_df[['st', 'ymdh']].apply(lambda x: str(x['st']) + '_' + str(x['ymdh']), axis=1)
model19_df = pd.merge(fy3d_df, fy4a_daily_df, on='st_time', suffixes=("", "_copy"))
generate_samples(model19_df, model19_cols_name, y_col_name, out_model19_samples_path, model_fix=19)

print('时间序列样本生成结束.')



