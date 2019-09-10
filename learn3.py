# -*- encoding: utf-8 -*-
"""
使用公式法：
theta = (X.T * X).I * X.T * Y
预测功率和电流之间的关系
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 解决画图中文显示问题
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# 读取文件
path1 = r'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)  # 数据使用;分割

# 异常数据的处理
new_df = df.replace('?', np.nan)  # 将 ? 的数据替换成 nan
datas = new_df.dropna(axis=0, how='any')  # 每一行数据中，如果存在一个特征值为nan，则整行删除

# 功率和电流之间的关系
X2 = datas.iloc[:, 2:4]
Y2 = datas.iloc[:, 5]

# 数据划分成训练集和测试集
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=0)

# 数据归一化
# scaler2 = StandardScaler()
# X2_train = scaler2.fit_transform(X2_train)
# X2_test = scaler2.transform(X2_test)

X = np.mat(X2_train)
Y = np.mat(Y2_train).reshape(-1, 1)  # reshape转为列向量，因为mat转换为矩阵后悔变成夯向量

theta = (X.T * X).I * X.T * Y
print(theta)

y_hat = np.mat(X2_test) * theta

# 绘制图表
t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u'线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()
