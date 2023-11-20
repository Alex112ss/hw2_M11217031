import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# 讀取數據
data = pd.read_csv('HousingData.csv')

# 分割特徵和目標變數
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# 初始化XGBoost模型
model = XGBRegressor()

# 訓練模型（使用所有特徵）
model.fit(X, y)

# 獲取特徵的重要性
feature_importance = model.feature_importances_

# 設定特徵重要性閾值，這裡設定為0.01
threshold = 0.01

# 選擇重要性大於閾值的特徵
selected_features = np.where(feature_importance > threshold)[0]

# 初始化K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存儲每個fold的表現
fold_scores_before = {'MAPE': [], 'RMSE': [], 'R2': []}
fold_scores_after = {'MAPE': [], 'RMSE': [], 'R2': []}

# 遍歷每個fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 訓練模型（使用所有特徵）
    model.fit(X_train, y_train)

    # 預測
    y_pred_before = model.predict(X_test)

    # 計算評估指標
    mape_before = np.mean(np.abs((y_test - y_pred_before) / y_test)) * 100
    rmse_before = np.sqrt(mean_squared_error(y_test, y_pred_before))
    r2_before = r2_score(y_test, y_pred_before)

    # 儲存評估指標
    fold_scores_before['MAPE'].append(mape_before)
    fold_scores_before['RMSE'].append(rmse_before)
    fold_scores_before['R2'].append(r2_before)

    # 訓練模型（使用特徵選擇後的特徵）
    model.fit(X_train.iloc[:, selected_features], y_train)

    # 預測
    y_pred_after = model.predict(X_test.iloc[:, selected_features])

    # 計算評估指標
    mape_after = np.mean(np.abs((y_test - y_pred_after) / y_test)) * 100
    rmse_after = np.sqrt(mean_squared_error(y_test, y_pred_after))
    r2_after = r2_score(y_test, y_pred_after)

    # 儲存評估指標
    fold_scores_after['MAPE'].append(mape_after)
    fold_scores_after['RMSE'].append(rmse_after)
    fold_scores_after['R2'].append(r2_after)

# 計算平均績效
average_scores_before = {metric: np.mean(scores) for metric, scores in fold_scores_before.items()}
average_scores_after = {metric: np.mean(scores) for metric, scores in fold_scores_after.items()}

# 輸出每個fold的預測績效以及5 folds的平均績效（特徵刪減前後）
print("Before feature selection:")
for i in range(1, kf.get_n_splits() + 1):
    print(f'Fold {i} - MAPE: {fold_scores_before["MAPE"][i-1]:.2f}%, RMSE: {fold_scores_before["RMSE"][i-1]:.2f}, R2: {fold_scores_before["R2"][i-1]:.4f}')
print(f'Average - MAPE: {average_scores_before["MAPE"]:.2f}%, RMSE: {average_scores_before["RMSE"]:.2f}, R2: {average_scores_before["R2"]:.4f}')

print("\nAfter feature selection:")
for i in range(1, kf.get_n_splits() + 1):
    print(f'Fold {i} - MAPE: {fold_scores_after["MAPE"][i-1]:.2f}%, RMSE: {fold_scores_after["RMSE"][i-1]:.2f}, R2: {fold_scores_after["R2"][i-1]:.4f}')
print(f'Average - MAPE: {average_scores_after["MAPE"]:.2f}%, RMSE: {average_scores_after["RMSE"]:.2f}, R2: {average_scores_after["R2"]:.4f}')

# 計算性能指標的差異
performance_diff = {metric: average_scores_after[metric] - average_scores_before[metric] for metric in average_scores_before}

print("\nPerformance Difference:")
for metric, diff in performance_diff.items():
    print(f'{metric}: {diff:.4f}')
