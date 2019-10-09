# -*- encoding: utf-8 -*-

# 为了解决过拟合问题：我们可以选择在损失函数中加入正则项（惩罚项），对于系数过大的惩罚
# 对于系数过多也有一定的惩罚能力，主要分为L1-norm 与 L2-norm
# LASSO 可以产生稀疏解，要用于特征选择（去掉冗余与无用的特征属性）


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 创建模型数据
np.random.seed(100)
# 显示方式设置，每次的字符数用于插入换行符，是否使用科学计数法
np.set_printoptions(linewidth=1000, suppress=True)

N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
# 将其设置为矩阵
x.shape = -1, 1
y.shape = -1, 1

# EidgeCV和Ridge的区别是：前者可以进行交叉验证
models = [
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', LinearRegression(fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        # alpha 给定的是Ridge算法中，L2正则项的权重值，也就是ppt中的兰木达
        # alphas是给定CV交叉验证过程中，Ridge算法的alpha参数值得取值范围
        ('Linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', LassoCV(alphas=np.logspace(0, 1, 10), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', ElasticNetCV(alphas=np.logspace(0, 1, 10), l1_ratio=[.1, .5, .7, .9, .95, 1]))
    ]),
]
