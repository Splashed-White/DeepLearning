'''
    打印MNIST数据集中图片的标签,独热编码表示
'''
# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 读取mnist数据集,如果不存在会事先下载
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
    1. X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，第二维中取第0个数据
    2. numpy.argmax(array, axis) array是矩阵，axis是0或者1;axis默认按列计算
       0表示的是按行比较返回最大值的索引，1表示按列比较返回最大值的索引
        eg: one_dim_array = np.array([1, 4, 5, 3, 7, 2, 6])
            print(np.argmax(one_dim_array))   # 4
'''
# 看前20张训练图片的label
for i in range(20):
    # 得到one-hot表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    one_hot_label = mnist.train.labels[i, :]
    # 通过np.argmax我们可以直接获得原始的label
    # 因为只有1位为1，其他都是0
    label = np.argmax(one_hot_label)
    print(f'mnist_train_{i}.jpg label: {label}')