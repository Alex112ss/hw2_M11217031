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

# 初始化K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存儲每個fold的表現
fold_scores = {'MAPE': [], 'RMSE': [], 'R2': []}

# 遍歷每個fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 訓練模型
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 計算評估指標
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 儲存評估指標
    fold_scores['MAPE'].append(mape)
    fold_scores['RMSE'].append(rmse)
    fold_scores['R2'].append(r2)

# 計算平均績效
average_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}

# 輸出每個fold的預測績效以及5 folds的平均績效
for i in range(1, kf.get_n_splits() + 1):
    print(f'Fold {i} - MAPE: {fold_scores["MAPE"][i-1]:.2f}%, RMSE: {fold_scores["RMSE"][i-1]:.2f}, R2: {fold_scores["R2"][i-1]:.4f}')

print(f'Average - MAPE: {average_scores["MAPE"]:.2f}%, RMSE: {average_scores["RMSE"]:.2f}, R2: {average_scores["R2"]:.4f}')
