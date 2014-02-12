#coding=utf8

"""
从训练集中生成两部分数据集，一部分用来做训练模型，另外一部分用来测试模型
"""
from random import random

def main():
    path = 'dataset/train2.csv'
    f = open(path, 'r')
    ftrain = open('dataset/train-model.csv', 'w')
    ftest = open('dataset/test-model.csv', 'w')
    for line in f:
        r = random()
        # 训练集：测试集 = 4：1
        if r < 0.2:
            ftest.write(line + '\n')
        else:
            ftrain.write(line + '\n')
            
    f.close()
    ftrain.close()
    ftest.close()

if __name__ == '__main__':
    main()
