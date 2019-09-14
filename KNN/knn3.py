# -*- encoding: utf-8 -*-
# 导入内置数据集模块
from sklearn import datasets
# 导入sklearn.neighbors模块中KNN类
from sklearn.neighbors import KNeighborsClassifier
# 用于训练集和测试机划分
from sklearn.model_selection import train_test_split


# 设置随机种子，不设置的话默认是按系统时间作为参数，因此每次调用随机模块时产生的随机数都不一样设置后每次产生的一样
random_state = 0
# 导入鸢尾花的数据集，iris是一个类似于结构体的东西，内部有样本数据，如果是监督学习还有标签数据
iris = datasets.load_iris()
# 样本数据150*4二维数据，代表150个样本，每个样本4个属性分别为花瓣和花萼的长、宽
iris_x = iris.data
# 长150的已为数组，样本数据的标签
iris_y = iris.target
iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_x, iris_y, random_state=random_state)

# 定义一个knn分类器对象
knn = KNeighborsClassifier()
# 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
knn.fit(iris_x_train, iris_y_train)
# 调用该对象的测试方法，主要接收一个参数：测试数据集
iris_y_predict = knn.predict(iris_x_test)
# 计算各测试样本基于概率的预测
probility = knn.predict_proba(iris_x_test)
# 计算与最后一个测试样本距离最近的5个点，返回的是这些样本的序号组成的数组
neighborpoint = knn.kneighbors(iris_x_test[-1].reshape(1, -1), 5, False)
# 调用该对象的打分方法，计算出准确率
score = knn.score(iris_x_test, iris_y_test, sample_weight=None)

# 输出测试的结果
print('iris_y_predict = ')
print(iris_y_predict)

# 输出原始测试数据集的正确标签，以方便对比
print('iris_y_test = ')
print(iris_y_test)

# 输出准确率计算结果
print('Accuracy: ', score)
print('neighborpoint of last test sample: ', neighborpoint)
print('probility: ', probility)
