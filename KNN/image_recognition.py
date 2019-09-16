# -*- encoding: utf-8 -*-
"""
利用KNN算法实现图像识别
数据源：The CIFAR-10 dataset
http://www.cs.toronto.edu/~kriz/cifar.html
"""
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_batch_1 = unpickle(file=r'../datas/cifar-10-batches-py/data_batch_1')
for k, v in data_batch_1.items():
    print(k, v)


# # 读取图片的数据，存放到img
# img = plt.imread('')
# # 打印图片的大小
# print(img.shape)
# # 展示图片
# plt.imshow(img)


