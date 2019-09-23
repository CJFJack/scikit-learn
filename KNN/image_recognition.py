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
