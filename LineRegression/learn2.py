# -*- encoding: utf-8 -*-
"""
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
X = datas.iloc[:, 2:4]
Y2 = datas.iloc[:, 5]

# 数据划分成训练集和测试集
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=0)

# 数据归一化
ss2 = StandardScaler()
X2_train = ss2.fit_transform(X2_train)
X2_test = ss2.transform(X2_test)

# 模型训练
lr2 = LinearRegression(fit_intercept=True)  # fit_intercept=True表示使用截距，即theta0，当数据进行归一化后，必须使用截距
lr2.fit(X2_train, Y2_train)

# 结果预测
Y2_predict = lr2.predict(X2_test)

# 模型评估
print('电流预测准确率：', lr2.score(X2_test, Y2_test))
print('电流参数：', lr2.coef_)
print('截距: ', lr2.intercept_)

# 模型的持久化，保存
from sklearn.externals import joblib
joblib.dump(ss2, r'result/data_ss2.model')
joblib.dump(lr2, r'result/data_lr2.model')

# 模型的加载
ss3 = joblib.load(r'result/data_ss2.model')
lr3 = joblib.load(r'result/data_lr2.model')
datas2 = [[23, 33]]
datas2 = ss3.transform(datas2)
print(lr3.predict(datas2))

# 绘制图表
t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u'线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
# plt.show()
