# -*- encoding: utf-8 -*-
"""
多项式扩展-预测时间和电压之间的关系
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

path1 = r'../datas/household_power_consumption_1000.txt'
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
Y = datas['Voltage']


# print(type(Y))
# print(Y.head(4))


def data_format(dt):
    # dt是Series [Date] [Time]
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


X = datas.iloc[:, 0:2]
X = X.apply(lambda x: pd.Series(data_format(x)), axis=1)
# print(type(X))
# print(X.head(4))

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

# 多项式扩展
models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),  # 多项式扩展
        ('Linear', LinearRegression(fit_intercept=False))  # 线性回归，不要截距
    ])
]
model = models[0]

# 训练模型
t = np.arange(len(X_test))  # 横坐标范围

N = 5
d_pool = np.arange(1, N, 1)  # 1-4阶
m = d_pool.size
clrs = []  # 颜色
for c in np.linspace(16711680, 255, m):
    clrs.append('#%06x' % int(c))
line_width = 3

plt.figure(figsize=(12, 6), facecolor='w')  # 创建一个绘图窗口，设置大小，设置颜色
for i, d in enumerate(d_pool):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(t, Y_test, 'k-', label=u'真实值', ms=10, zorder=N)
    # 设置管道对象中的参数值，Poly是在管道对象中定义的操作名称，后面跟参数名称；中间是两个下划线
    model.set_params(Poly__degree=d)  # 设置多项式的阶乘
    model.fit(X_train, Y_train)  # 模型训练
    # Linear是管道中定义的操作名称
    # 获取线性回归算法模型对象
    lin = model.get_params()['Linear']
    output = u'%d阶，系数为：' % d
    # 判断lin对象中是否有对应的属性
    if hasattr(lin, 'alpha_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'alpha=%.6f. ' % lin.alpha_) + output[idx:]
    if hasattr(lin, 'l1_ratio_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
    print(output, lin.coef_.ravel())

    # 模型结果预测
    y_hat = model.predict(X_test)
    # 计算评估值
    s = model.score(X_test, Y_test)

    # 画图
    z = N - 1 if d == 2 else 0
    label = u'%d阶，准确率=%.3f' % (d, s)
    plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha=0.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

# 预测值和实际值画图比较
plt.suptitle(u'线性回归预测时间和电压之间的多项式关系', fontsize=20)
plt.grid(b=True)
plt.show()
