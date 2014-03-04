#coding=utf8

"""
Task1: Multi-label task
"""

import numpy as np

from dimension_reduction import PCA_transform
from evaluation import predition2dict
from evaluation import score_list_v2 as caculate_AP_dict

def average_precision(outputs, test_target):
    """ Example-based AP for multilable classification
    outputs: shape=n, num_class, probility prediction
    test_targe: the same size of outputs, 0/1 indicates whether belong to the class
    """
    assert(outputs.shape == test_target.shape)
    num_instance, num_class = outputs.shape
    
    aveprec = 0.0
    num_valid_instance = 0 # 记录那些至少属于一个类的样本个数
    for i in range(num_instance):
        # sort prediction prob. 
        tmp = outputs[i, :]
        # decending order
        index = list(np.argsort(-1 * tmp)) # [tempvalue,index] = sort(temp)
        
        indicator = np.array([0] * num_class, int)
        # Which and how many classes does this sample belong to ?
        label_list = np.array(range(num_class))[test_target[i, :] == 1]
        label_size = len(label_list)
        for m in range(label_size):
            label = label_list[m]
            loc = index[label]
            indicator[loc] = 1
            
        summary = 0
        for m in range(label_size):
            label = label_list[m]
            loc = index[label]
            #summary = summary + sum(indicator[loc:num_class]) * 1.0 / (num_class-loc+1)
            summary = summary + sum(indicator[:loc+1]) * 1.0 / (loc+1)
        
        # 不考虑那些不属于任何一类的样本    
        if label_size > 0:
            aveprec = aveprec + summary / label_size
            num_valid_instance += 1

    Average_Precision = aveprec / num_valid_instance
    
    return Average_Precision

def OneVsRest_multilabel(train_feature, train_label, test_feature, BinaryClassifier, **kwargs):
    """ multi-label classification
    """
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier(BinaryClassifier(**kwargs)).fit(train_feature, train_label)
    
    train_pred = clf.predict_proba(train_feature)
    test_pred = clf.predict_proba(test_feature)
        
    return train_pred, test_pred
    
def save_multilabel_result(method_name, n_components, model_test_AP, test_pred):
    """ Save result to file
    """
    # save the final prediction
    index = 0
    path = 'results/task1-%s-%d-%f-.csv' % (method_name, n_components, model_test_AP)
    print 'Saving result to file: ', path
    f = open(path, 'w')
    n = len(test_pred)
    for i in range(n):
        result = ','.join(str(x) for x in test_pred[i, :])
        f.write('%d,%s\n' % (index+1, result))
        index += 1
        
    f.close()

if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    n_folds = int(sys.argv[2])
    
    # load dataset
    f = open('task1-dataset/task1-dataset.pickle')
    import pickle
    # the following variables are accessed throughout this script, be careful
    train_feature, train_label, test_feature = pickle.load(f)
    f.close()
    
    # apply the PCA dimension reduction
    train_feature, test_feature = PCA_transform(train_feature, test_feature, 'task1-dataset/task1-PCA-decomp.mat')
    train_feature = train_feature[:, :n_components]
    test_feature = test_feature[:, :n_components]
    
    train_count, num_class = train_label.shape
    
    from sklearn.naive_bayes import GaussianNB as BinaryClassifier
    method_name = 'GaussianNB+OneVsRest'
    kwargs = {}
    
    #from sklearn.svm import SVC as BinaryClassifier
    #method_name = 'SVC+OneVsRest'
    #kwargs = {'C':5, 'gamma':0.05, 'probability':'True'}
    
    #method_name = 'kNN+OneVsRest'
    #from sklearn.neighbors import KNeighborsClassifier as BinaryClassifier
    #kwargs = {'k':1}
    
    print 'Method: ', method_name
    from sklearn.cross_validation import KFold
    kf = KFold(len(train_label), n_folds, indices=True)
    index = 0
    model_test_AP = [0] * n_folds
    for train_index, test_index in kf:
        print 'Prepare cv dataset: %d' % index
        model_train_feature = train_feature[train_index, :]
        model_test_feature = train_feature[test_index, :]
        model_train_label = train_label[train_index, :]
        model_test_label = train_label[test_index, :]
        
        print 'Train classifiers using the CLR method...'
        train_pred, test_pred = OneVsRest_multilabel(model_train_feature, model_train_label, model_test_feature, BinaryClassifier, **kwargs)
        
        #import ipdb; ipdb.set_trace()
        model_test_AP[index] = average_precision(test_pred, model_test_label)
        # use the evaluation metric provided by CCDM2014 host
        #model_test_AP[index] = caculate_AP_dict(predition2dict(test_pred), predition2dict(model_test_label))
        print 'Model testing AP:', model_test_AP[index]
        
        index += 1
    
    avg_AP = sum(model_test_AP) / n_folds
    print '\nAverage AP:', avg_AP
    
    print 'Train the whole dataset...'
    train_pred, test_pred = OneVsRest_multilabel(train_feature, train_label, test_feature, BinaryClassifier, **kwargs)
    save_multilabel_result(method_name, n_components, avg_AP, test_pred)
    
    
