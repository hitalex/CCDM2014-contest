#coding=utf8

"""
生成随机结果
"""

import numpy as np

def gen_random_multiclass_results(train_feature, train_label, test_feature):
    """ Generate random multi-class prediction results
    """
    train_count = len(train_label)
    test_count = len(test_feature)
    num_class = 3
    class_count = np.array([0] * num_class, int)
    for i in range(train_count):
        class_count[train_label[i]] += 1
        
    prior = class_count * 1.0 / train_count
    
    test_pred = np.random.choice(range(num_class), size = test_count, p = prior)
        
    # save the final prediction
    index = 0
    f = open('results/task2-random-.csv', 'w')
    for y in test_pred:
        f.write('%d,%d\n' % (index+1, test_pred[index]))
        index += 1
        
    f.close()
    
if __name__ == '__main__':
    print 'Load dataset...'
    import pickle
    f = open('task2-dataset/task2-dataset.pickle', 'r')
    train_feature, train_label, test_feature = pickle.load(f)
    f.close()
    
    gen_random_multiclass_results(train_feature, train_label, test_feature)
