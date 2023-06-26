import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# 存储所有样本的电子信号和生命体征数据
opticalpower_data = []
breath_data = []
heart_rate_data = []
totalMotion_data = []

# 读取所有数据文件并提取数据
data_dir = r"D:\DJob\C2988\train"
model_dir = r"D:\DJob\C2988\linear_model\model"
# count = 0
for filename in os.listdir(data_dir):
    # if count > 500:
    #     break
    # count += 1
    with open(os.path.join(data_dir, filename), "r") as file:
        data = json.load(file)
        opticalpower_data.append(data["opticalpower"])
        breath_data.append(data["breath"])
        heart_rate_data.append(data["heart_rate"])
        totalMotion_data.append(data["totalMotion"])

# 将数据转换为NumPy数组
opticalpower_data = np.array(opticalpower_data)
breath_data = np.array(breath_data)
heart_rate_data = np.array(heart_rate_data)
totalMotion_data = np.array(totalMotion_data)


# 创建保存模型的路径
os.makedirs(model_dir, exist_ok=True)

# 使用线性回归模型拟合生命体征数据
regressor_breath = LinearRegression()
regressor_breath.fit(opticalpower_data, breath_data)
joblib.dump(regressor_breath, os.path.join(model_dir, "regressor_breath.pkl"))
print("regressor_breath模型已保存到", model_dir, "目录下")

regressor_heart_rate = LinearRegression()
regressor_heart_rate.fit(opticalpower_data, heart_rate_data)
joblib.dump(regressor_heart_rate, os.path.join(model_dir, "regressor_heart_rate.pkl"))
print("regressor_heart_rate模型已保存到", model_dir, "目录下")

regressor_totalMotion = LinearRegression()
regressor_totalMotion.fit(opticalpower_data, totalMotion_data)
joblib.dump(regressor_totalMotion, os.path.join(model_dir, "regressor_totalMotion.pkl"))
print("regressor_totalMotion模型已保存到", model_dir, "目录下")

# 计算训练集上的评估指标
train_predictions_breath = regressor_breath.predict(opticalpower_data)
train_predictions_heart_rate = regressor_heart_rate.predict(opticalpower_data)
train_predictions_totalMotion = regressor_totalMotion.predict(opticalpower_data)

mse_breath = mean_squared_error(breath_data, train_predictions_breath)
mae_breath = mean_absolute_error(breath_data, train_predictions_breath)
rmse_breath = np.sqrt(mse_breath)

mse_heart_rate = mean_squared_error(heart_rate_data, train_predictions_heart_rate)
mae_heart_rate = mean_absolute_error(heart_rate_data, train_predictions_heart_rate)
rmse_heart_rate = np.sqrt(mse_heart_rate)

mse_totalMotion = mean_squared_error(totalMotion_data, train_predictions_totalMotion)
mae_totalMotion = mean_absolute_error(totalMotion_data, train_predictions_totalMotion)
rmse_totalMotion = np.sqrt(mse_totalMotion)

print("训练集上的评估指标：")
print("呼吸频率：")
print("平均绝对误差（MAE）：", mae_breath)
print("均方根误差（RMSE）：", rmse_breath)

print("心率：")
print("平均绝对误差（MAE）：", mae_heart_rate)
print("均方根误差（RMSE）：", rmse_heart_rate)

print("体动：")
print("平均绝对误差（MAE）：", mae_totalMotion)
print("均方根误差（RMSE）：", rmse_totalMotion)




