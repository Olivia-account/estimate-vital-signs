import json
import numpy as np
import joblib

# 加载模型
regressor_breath = joblib.load(r"D:\DJob\C2988\SVR\model\regressor_breath.pkl")
regressor_heart_rate = joblib.load(r"D:\DJob\C2988\SVR\model\regressor_heart_rate.pkl")
regressor_totalMotion = joblib.load(r"D:\DJob\C2988\SVR\model\regressor_totalMotion.pkl")

# 手动输入数据
opticalpower = [float(x) for x in input("请输入电子信号数据（以空格分隔）：").split()]

# 将数据转换为NumPy数组并进行预测
opticalpower_data = np.array([opticalpower])
predicted_breath = regressor_breath.predict(opticalpower_data)
predicted_heart_rate = regressor_heart_rate.predict(opticalpower_data)
predicted_totalMotion = regressor_totalMotion.predict(opticalpower_data)

# 打印预测结果
print("预测的呼吸频率:", predicted_breath[0])
print("预测的心率:", predicted_heart_rate[0])
print("预测的体动:", predicted_totalMotion[0])
