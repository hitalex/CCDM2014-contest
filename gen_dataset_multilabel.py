#coding=utf8

"""
Generate dataset for task2
"""

from random import random

import numpy as np

def split_model_dataset(train_data):
    """ split model training and test dataset
    """
    n = len(train_data)
    threshold = 1/3.5 # 1:3.5, just as the original test:train
    # test and train row index 
    test_index = []
    train_index = []
    for i in range(n):
        r = random()
        # 训练集：测试集 = 4：1
        if r < threshold:
            test_index.append(i)
        else:
            train_index.append(i)
    
    model_test_feature = train_data[test_index, :129]
    model_test_label = train_data[test_index, 129:]
    
    model_train_feature = train_data[train_index, :129]
    model_train_label = train_data[train_index, 129:]
    
    return model_train_feature, model_train_label, model_test_feature, model_test_label

def main():
    # all features are nominal: 0 or 1
    train_data = np.genfromtxt('task1-dataset/train_data.csv', dtype=int, delimiter=',')
    test_feature = np.genfromtxt('task1-dataset/test_feature_data.csv', dtype=int, delimiter=',')
    
    # remove the first line
    test_feature = test_feature[:, 1:]
    # prepare the numpy format
    train_label = train_data[:, 129:]
    train_label[train_label == -1] = 0 # zero elements indicate do not belong to the class
    
    # statics of each class
    train_count, num_class = train_label.shape
    print 'Total count:', train_count, 'Number of class:', num_class
    p = np.array([0] * num_class)
    # num_class_membership[i] 存储属于i个class的样本个数
    num_class_membership = [0] * (num_class+1)
    for i in range(train_count):
        sample = train_label[i, :]
        p[sample == 1] += 1
        # 查看该样本属于多少个类
        n = len(p[sample == 1])
        num_class_membership[n] += 1
        
    print 'Number of samples in each class:', p
    print '属于i个类的样本个数：', num_class_membership
    
    model_train_feature, model_train_label, model_test_feature, model_test_label = split_model_dataset(train_data)
    
    f = open('task1-dataset/task1-dataset.pickle', 'w')
    import pickle
    pickle.dump([model_train_feature, model_train_label, model_test_feature, model_test_label, test_feature], f)
    f.close()
    
if __name__ == '__main__':
    main()
