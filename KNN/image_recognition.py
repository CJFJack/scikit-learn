# -*- encoding: utf-8 -*-
"""
利用KNN算法实现图像识别
数据源：The CIFAR-10 dataset
http://www.cs.toronto.edu/~kriz/cifar.html

"""

"""
1. 文件的读取、可视化、以及采样
在这部分我们需要读取图片文件，并展示部分图片便于观察，以及做少量的采样。

文件的读取： 读取部分的代码已经提供，你只需要调用一下即可以读取图片数据。
可视化： 选择其中的一些样本做可视化，也就是展示图片的内容以及它的标签。
采样：统计一下各类出现的个数以及采样部分样本作为后续的模型的训练。
"""
# 文件的读取，我们直接通过给定的`load_CIFAR10`模块读取数据。
from load_data import load_CIFAR10  # 感谢这个magic函数，你不必要担心如何写读取的过程。如果想了解细节，可以参考此文件。
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV  # 通过网格方式来搜索参数

cifar10_dir = '../datas/cifar-10-batches-py'  # 定义文件夹的路径：请不要修改此路径！ 不然提交后的模型不能够运行。

# 读取文件，并把数据保存到训练集和测试集合。
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 先来查看一下每个变量的大小，确保没有任何错误！X_train和X_test的大小应该为 N*W*H*3
# N: 样本个数, W: 样本宽度 H: 样本高度， 3: RGB颜色。 y_train和y_test为图片的标签。
print("训练数据和测试数据:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print("标签的种类: ", np.unique(y_train))  # 查看标签的个数以及标签种类，预计10个类别。

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)  # 样本种类的个数
samples_per_class = 5  # 每一个类随机选择5个样本

# TODO 图片展示部分的代码需要在这里完成。 hint:  plt.subplot函数以及 plt.imshow函数用来展示图片
for idx, cls in enumerate(classes):
    ran_list = random.sample(list(np.argwhere(y_train == idx)[:, 0]), samples_per_class)
    for r_idx, r in enumerate(ran_list):
        plt.subplot(5, 10, idx + 1 + (r_idx * 10))
        plt.imshow(X_train[r, :, :, :] / 255)
        plt.axis('off')
        if r_idx == 0:
            plt.title(cls)

plt.show()

# TODO 统计并展示每一个类别出现的次数
class_dict = defaultdict(int)
for y in y_train:
    class_dict[classes[y]] += 1
print(dict(class_dict))

# TODO 随机采样训练样本5000个和测试样本500个。训练样本从训练集里采样，测试样本从测试集里采样。
num_training = 5000
num_test = 500

train_row_rand_array = np.arange(X_train.shape[0])
np.random.shuffle(train_row_rand_array)
X_train = X_train[train_row_rand_array[0:num_training], :, :, :]
y_train = y_train[train_row_rand_array[0:num_training]]

test_row_rand_array = np.arange(X_test.shape[0])
X_test = X_test[test_row_rand_array[0:num_test], :, :, :]
y_test = y_test[test_row_rand_array[0:num_test]]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

"""
2. 使用KNN算法识别图片。
这部分主要的工作是通过K折交叉验证来训练KNN，以及选择最合适的K值和p值。
具体KNN的描述请看官方文档： https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

KNN有几个关键的参数：

K: 指定选择多少个neighbors。 这个值越大，我们知道KNN的决策边界就会越平滑，而且越不容易过拟合， 但不保证准确率会很高。
p: 不同距离的指定，看以下的说明。

KNN依赖于两个样本的距离计算，这里简单介绍一下一个概念叫做Minkowski Distance，是一个种通用的距离计算方法。
假如我们有两个点，分别由两个向量来表达 x=(x1,x2,...,xd) 和 y=(y1,y2,...,yd) ，这时候根据Minkowski Distance的定义可以得到以下的结果：

dist(x,y)=(∑i~d|xi−yi|^p)^(1/p)
 
从上述的距离来看其实不难发现 p=1 时其实就是绝对值的距离， p=2 时就是欧式距离。
所以欧式距离其实是Minkowski Distance的一个特例而已。所以这里的 p 值是可以调节的比如 p=1,2,3,4,... 。

"""
# 首先我们 Reshape一下图片。图片是的每一个图片变成一个向量的形式。也就是把原来大小为(32, 32, 3)的图片直接转换成一个长度为32*32*3=3072的向量。
# 这样就直接可以作为模型的输入。 X_train_1和y_train_1是用来解决第一个问题的处理后的数据。
X_train1 = np.reshape(X_train, (X_train.shape[0], -1))
X_test1 = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train1.shape, X_test1.shape)  # 确保维度正确

# 使用K折交叉验证去训练最好的KNN模型，并给出最好的交叉验证的结果（准确率）和在测试集上的准确率。
# 需要搜索的参数为K和p。对于交叉验证，在这里使用GridSearchCV,这是一种参数搜索的方法也叫作网格搜索，
# 其实就是考虑所有的组合。 比如K=[1,3,5,7], p=[1,2,3], 则通过网格搜索会考虑所有可能的 12 种组合。
# TODO 通过K折交叉验证构造最好的KNN模型，并输出最好的模型参数，以及测试集上的准确率。
# 训练数据： （X_train1, y_train）, 测试数据：(X_test1, y_test)
params_k = [1, 3, 5, 7, 9, 11, 13]  # 可以选择的K值
params_p = [1, 2, 3]  # 可以选择的P值

# 构建模型
parameters = {'n_neighbors': params_k}
knn = KNeighborsRegressor()
model = GridSearchCV(knn, parameters, cv=5)
model.fit(X_train1, y_train)

# 输出最好的K和p值
print(model.best_params_)
# 输出训练集分数
print(model.best_score_)

# 输出在测试集上的准确率
knn_clf = model.best_estimator_
y_pre = knn_clf.predict(X_test1)
print(knn_clf.score(X_test1, y_pre))

"""
3. 抽取图片特征，再用KNN算法来识别图片
在课程里也讲过，一种解决图像识别问题中各种环境不一致的方案是抽取对这些环境因素不敏感的特征。这就是所谓的特征工程。 在这里，我们即将会提取两种类型的特征，分别是color histogram和HOG特征，并把它们拼接在一起作为最终的特征向量。 至于这些特征的概念请参考第三章的内容，或者网络上的一些解释。我们已经提供了抽取特征的工具，只需要调用就可以使用了。 所以你需要做的任务是：

调用特征提取工具给每一个图片提取特征。 如果想深入了解，可以查看其代码
使用K折交叉验证去学出最好的模型（同上）
"""

from features import *

num_color_bins = 10 # 设定color histogram的 bin大小

# 分别设置接下来需要调用的两个特征抽取器，分别是hog_feature, color_histogram_hsv
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]

# 抽取特征，分别对特征数据和测试数据，把结果存放在X_train2和X_test2
X_train2 = extract_features(X_train, feature_fns, verbose=True)
#X_val_feats = extract_features(X_val, feature_fns)
X_test2 = extract_features(X_test, feature_fns)

# 打印转换后的数据大小。
print (X_train2.shape, X_test2.shape)