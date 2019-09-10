# -*- encoding: utf-8 -*-
"""
预测时间和有用功率之间的关系
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

path1 = r'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)
# print(type(df))
# print(df.index)
# print(df.columns)
# print(df.head(3))
# print(df.info())

# 异常数据的处理
new_df = df.replace('?', np.nan)  # 将 ? 的数据替换成 nan
datas = new_df.dropna(axis=0, how='any')  # 每一行数据中，如果存在一个特征值为nan，则整行删除

# print(datas.index)
# print(datas.columns)
# print(datas.describe().T)

# print(datas)
Y = datas['Global_active_power']
# print(type(Y))
print(Y.head(4))


def data_format(dt):
    # dt是Series [Date] [Time]
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


X = datas.iloc[:, 0:2]
X = X.apply(lambda x: pd.Series(data_format(x)), axis=1)
# print(type(X))
print(X.head(4))

# 对数据进行测试集、训练集划分
# X: 特征矩阵(类型一般是DataFrame)
# Y: 特征对应的Label标签或目标属性(类型一般是Series)

# test_size=0.2表示将数据集的百分之20%划分为测试集
# 如果test_size和train_size都不设置，则默认划分25%为测试集，剩下为训练集
# random_state=0表示随机种子等于0，默认是随机数，设置随机种子为了保证每次运行程序划分的训练集和测试集结果保持一样
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train.shape)
# print(Y_test.shape)

# # 随机种子的作用
# np.random.seed(7)
# arr1 = np.random.randint(1, 11, (10, ))
# print(arr1)
# np.random.seed(7)
# arr2 = np.random.randint(1, 11, (10, ))
# print(arr2)

# print(X_train.describe().T)

ss = StandardScaler()
# 训练加转化
X_train = ss.fit_transform(X_train)
# 直接转化
X_test = ss.transform(X_test)
# print(type(X_train))
# print(pd.DataFrame(X_train).describe().T)

# fit_intercept=True表示要加上特征系数w0
# normalize=False表示不使用内置的数据标准化模块
# copy_X=True表示不改变原数据
# n_jobs表示并行任务数
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, Y_train)
y_predict = lr.predict(X_test)

# R^2代表预测的准确率
print('训练集上的R^2：', lr.score(X_train, Y_train))
print('测试集上的R^2：', lr.score(X_test, Y_test))

# mse = np.average((y_predict - Y_test) ** 2)
# rmse = np.sqrt(mse)
# print('rmse: ', rmse)

print('模型训练后的系数：', end='')
print(lr.coef_)

print('模型的截距:', lr.intercept_)

t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')

plt.legend(loc='upper left')
plt.title('线性回归预测时间和功率之间的关系', fontsize=20)
# 网格
plt.grid(b=True)
plt.show()
