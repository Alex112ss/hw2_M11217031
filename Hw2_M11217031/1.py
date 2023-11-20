import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
import time
from math import sqrt
import numpy as np

# 加載數據，替換 'path/to/' 為實際的文件路徑
df_train = pd.read_csv('adult.data')
df_test = pd.read_csv('adult.test')

# 提取特徵和目標列
X_train = df_train.drop("hours-per-week", axis=1)
y_train = df_train["hours-per-week"]

X_test = df_test.drop("hours-per-week", axis=1)
y_test = df_test["hours-per-week"]

# 識別非數值型的列
categorical_cols = [cname for cname in X_train.columns if 
                    X_train[cname].dtype == "object"]

# 使用獨熱編碼處理非數值型的列
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# 初始化KNN回歸模型
knn_model = KNeighborsRegressor(n_neighbors=10)  # 調整鄰居數量

# 初始化SVR模型
svr_model = SVR(C=1.0, epsilon=0.1)  # 調整C和epsilon

# 初始化RandomForest回歸模型
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15)  # 調整n_estimators和max_depth

# 初始化XGBoost回歸模型
xgb_model = XGBRegressor(n_estimators=300, max_depth=10)  # 調整n_estimators和max_depth

# 所有模型的訓練和預測
models = [knn_model, svr_model, rf_model, xgb_model]
model_names = ['KNN', 'SVR', 'RandomForest', 'XGBoost']

for model, name in zip(models, model_names):
    # 訓練模型並計算時間
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # 在測試集上進行預測，並計算預測時間
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # 評估模型性能
    mse = mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 輸出結果
    print(f"{name}模型訓練時間：{training_time:.4f}秒")
    print(f"{name}模型預測時間：{prediction_time:.4f}秒")
    print(f"{name}均方誤差（MSE）：{mse:.4f}")
    print(f"{name}平均絕對百分比誤差（MAPE）：{mape:.4f}%")
    print(f"{name}均方根誤差（RMSE）：{rmse:.4f}")
    print(f"{name}R-squared（R2）：{r2:.4f}")
    print("\n")
