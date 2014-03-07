#coding=utf8

"""
Task2：
由csv文件生成ARFF文件的文件头信息
"""

import numpy as np
from numpy import genfromtxt

def main():
    train_path = '/home/kqc/dataset/CCDM2014/task2/train_data.csv'
    test_path = '/home/kqc/dataset/CCDM2014/task2/test_feature_data.csv'
    
    train_data = genfromtxt(train_path, delimiter=',')
    test_data = genfromtxt(test_path, delimiter=',')
    
    arff_file = open('task2-dataset/CCDM2014-contest-Task2-header', 'w')
    
    arff_file.write('@RELATION CCDM2014-Task2\n\n')
    
    # 删除test data中的第一列，此列为样本的序号
    test_data = test_data[:, 1:]
    label_train = train_data[:, -1] # 训练集最后一列为标签信息
    train_data = train_data[:, :-1]
    train_count, num_feature = train_data.shape
    # 将训练集和测试集合并
    whole_data = np.vstack((train_data, test_data))
    total, num_feature = whole_data.shape
    
    # 遍历所有的特征
    nominal_feature_index = []
    for i in range(num_feature):
        s = whole_data[:, i]
        """
        # 特殊处理第400个和第407个特征
        if i == 400:
            s = s / 25
            whole_data[:, i] = s
        elif i == 407:
            s = s / 20
            whole_data[:, i] = s
        """
        flag = False
        for j in range(total):
            if s[j] < 0 or int(s[j]) != s[j]:
                # 此列为numeric特征，可忽略
                flag = True
                break
        # write attribute info
        if flag:
            arff_file.write('@ATTRIBUTE f%d NUMERIC\n' % i)
            continue
        # s中只有0，不考虑
        elif max(s) == 0:
            arff_file.write('@ATTRIBUTE f%d NUMERIC\n' % i)
            continue
        # 如果该特征只有0和1两个取值，则同样不考虑
        elif min(s) == 0 and max(s) == 1:
            arff_file.write('@ATTRIBUTE f%d {0,1}\n' % i)
            continue
        else:
            nominal_feature_index.append(i)
            s = np.array(s, int)
            attr_values = np.unique(s)
            attr_values = [str(v) for v in attr_values]
            nominal_string = ','.join(attr_values)
            arff_file.write('@ATTRIBUTE f%d {%s}\n' % (i, nominal_string))
            
    # the class information
    arff_file.write('@ATTRIBUTE class {0,1,2}\n\n')
    arff_file.write('@DATA\n')
    arff_file.close()

if __name__ == '__main__':
    main()
