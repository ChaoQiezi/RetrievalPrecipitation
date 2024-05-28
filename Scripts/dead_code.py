# @Author   : ChaoQiezi
# @Time     : 2024/5/16  17:56
# @FileName : dead_code.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import joblib
import h5py


h5_path = r'H:\Datasets\Objects\retrieval_prcp\Data\LSTM\Samples\model19_train_test.h5'
scalers_path = r'I:\PyProJect\RetrievalPrecipitation\scalers.pkl'

with h5py.File(h5_path) as f:
    test_x = f['test_x'][0, :, :]
    test_y = f['test_y'][0, :]
    test_ix = f['test_ix'][0, :]
scalers = joblib.load(scalers_path)
x_scaler = scalers['model19_x_scaler']
y_scaler = scalers['model19_y_scaler']
test_x2 = x_scaler.inverse_transform(test_x)
test_y2 = y_scaler.inverse_transform([test_y])
