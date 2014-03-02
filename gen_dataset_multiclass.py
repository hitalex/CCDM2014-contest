#coding=utf8

"""
从训练集中生成两部分数据集，一部分用来做训练模型，另外一部分用来测试模型
"""
from random import random

import numpy as np

def main():
    train_path = 'task2-dataset/train2.csv'
    test_path = 'task2-dataset/test2.csv'
    
    train_data = np.genfromtxt(train_path, dtype=float, delimiter=',')
    test_feature = np.genfromtxt(test_path, dtype=float, delimiter=',')
    
    # the instance index is already removed
    test_feature = test_feature[:, :]
    # prepare the numpy format
    train_feature = train_data[:, :-1]
    train_label = np.array(train_data[:, -1], int)
    
    f = open('task2-dataset/task2-dataset.pickle', 'w')
    import pickle
    pickle.dump([train_feature, train_label, test_feature], f)
    f.close()

if __name__ == '__main__':
    main()
