# @Author   : ChaoQiezi
# @Time     : 2024/5/15  19:28
# @FileName : model_eval.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于模型评估/预测
"""


# 使用训练好的模型进行预测
# DataLoader
test_ds = TensorDataset(test_x, test_y)
test_data_loader = DataLoader(test_ds, batch_size=16)
predictions, real_labels = [], []
model.eval()
with torch.no_grad():
    for input, labels in test_data_loader:
        predicted = model(input.to(DEVICE))

        predictions.append(predicted.detach().cpu().numpy())
        real_labels.append(labels.detach().cpu().numpy())

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error

predictions = np.concatenate(predictions, axis=0)
real_labels = np.concatenate(real_labels, axis=0)


reals = []
preds = []
for row in range(predictions.shape[0]):
    pred = predictions[row, :]
    real = real_labels[row, :]

    reals.append(real[0])
    preds.append(pred[0])
    # preds.append(pred[0] if pred[0] > 0 else 0)
print('mse', mean_squared_error(reals, preds))
# print('rmse', mean_squared_log_error(reals, preds))
print('mae', mean_absolute_error(reals, preds))
print('r2', r2_score(reals, preds))
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
ax1 = axs[0]
ax2 = axs[1]
ax1.plot(reals)
ax2.plot(preds)
plt.show()