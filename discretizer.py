#coding=utf8

"""
功能：将task2中的numeric特征变换为nominal特征（f13除外）
输入：原始数据及经验数据
输出：ARFF header file, Train data, test data
转换方法基于以下观察：如果将task2中的每个numeric特征进行排序，发现它们成等差数列，
而且所有可能的取值数目并不是很多（相对与样本数来说）。所以根据此特点可以将其分隔离散化。
"""

import numpy as np
from numpy import genfromtxt

def numericlist2str(a):
    """ 将实数列表转换为字符串
    """
    a = list(a)
    n = len(a)
    for i in range(n):
        # 如果是整数，则存储为为不带小数点的模式
        if a[i] == int(a[i]):
            a[i] = str(int(a[i]))
        else:
            a[i] = str(a[i])
    
    s = '{' + ','.join(a) + '}'
    
    return s
    
def numeric2nominal(a):
    """ 将实数特征转换为字符串特征
    @a: 所有的实数数组
    Return:
    feature_desc: 特征描述列表
    new_str_feature: 新的特征描述
    """
    sv = list(np.sort(np.unique(a))) # 排序后的数字表示方法
    n = len(sv)
    feature_desc = [""] * n
    for i in range(n):
        if i == 0:
            feature_desc[i] = '(-inf~' + str(sv[0]) + ']'
        else:
            feature_desc[i] = '(' + str(sv[i-1]) + '~' + str(sv[i]) + ']'
            
    n = len(a)
    new_str_feature = [""] * n
    for i in range(n):
        index = sv.index(a[i])
        new_str_feature[i] = feature_desc[index]
        
    return feature_desc, new_str_feature
    
def write_header_file(feature_desc, base_path):
    num_feature = len(feature_desc)
    # 写入ARFF header file
    fheader = open(base_path + 'task2-discretize-header', 'w')
    fheader.write('@RELATION CCDM2014-Task2-Discretize\n\n')
    for i in range(num_feature):
        fheader.write('@ATTRIBUTE f%d %s\n' % (i, feature_desc[i]))
        
    fheader.write('@ATTRIBUTE class {0,1,2}\n\n')
    fheader.write('@DATA\n')
    fheader.close()

def main():
    train_path = '/home/kqc/dataset/CCDM2014/task2/train_data.csv'
    test_path = '/home/kqc/dataset/CCDM2014/task2/test_feature_data.csv'
    
    # 以字符串的形式读入矩阵
    train_data = genfromtxt(train_path, object, delimiter=',')
    test_data = genfromtxt(test_path, object, delimiter=',')
    
    # 删除test data中的第一列，此列为样本的序号
    test_data = test_data[:, 1:]
    label_train = train_data[:, -1] # 训练集最后一列为标签信息
    train_data = train_data[:, :-1]
    train_count, num_feature = train_data.shape
    # 将训练集和测试集合并
    whole_data = np.vstack((train_data, test_data))
    total, num_feature = whole_data.shape
    
    # 已经得到的numeric feature列表
    numeric_feature_list = [0, 11, 12, 13, 55, 379, 380, 382, 383, 384, 385, 386, \
        387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 401, 402, 403, 406]
    numeric_feature_set = set(numeric_feature_list)
    
    # 离散化的最大分支设定，如果不同的取值数超过此值，则不转换
    max_split = 50
    num_feature = 410 # 不包含class属性
    feature_desc = [''] * num_feature # 保存对该特征的描述，如'{1,2,3}'或'NUMERIC'，最终会写入到ARFF file header
    
    reduced_numeric_feature_list = [] # 最终的numeric特征列表
    for i in range(num_feature):
        s = np.unique(whole_data[:, i])
        if i in numeric_feature_set:
            if len(s) > max_split:
                feature_desc[i] = 'NUMERIC'
                reduced_numeric_feature_list.append(i)
            else:
                f = np.array(whole_data[:, i], float) # 原特征的数字表示方法
                f_desc_list, new_str_feature = numeric2nominal(f)
                feature_desc[i] = '{' + ','.join(f_desc_list) + '}'
                #print 'Feature:', i
                #print feature_desc[i]
                whole_data[:, i] = new_str_feature # 重新给出nominal赋值
        else:
            feature_desc[i] = '{' + ','.join(s) + '}' # 对nominal特征进行处理
            #print feature_desc[i]
    
    whole_label = np.concatenate((label_train, ['0'] * (total-train_count)))
    whole_label = np.array([whole_label]).T
    whole_data = np.hstack((whole_data, whole_label))
    print 'List of remaining numeric attributes:', reduced_numeric_feature_list
    base_path = '/home/kqc/dataset/CCDM2014/task2/discretize/'
    # write header files
    print 'Writing ARFF header files...'
    write_header_file(feature_desc, base_path)
    
    print 'Writing train and test data csv file...'
    np.savetxt(base_path + 'train-data.csv', whole_data[:train_count, :], fmt='%s', delimiter=',')
    np.savetxt(base_path + 'test-data.csv', whole_data[train_count:, :], fmt='%s', delimiter=',')
    
if __name__ == '__main__':
    main()
