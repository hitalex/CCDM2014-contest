#coding=utf8

"""
Task1: Multi-label task
"""

import numpy as np

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
        index = list(np.argsort(tmp)) # [tempvalue,index] = sort(temp)
        
        indicator = np.array([0] * num_class, int)
        # Which and how many classes does this sample belong to ?
        label_list = np.array(range(num_class))[test_target[i, :] == 1]
        label_size = len(label_list)
        for m in range(label_size):
            label = label_list[m]
            loc = index.index(label)
            indicator[loc] = 1
            
        summary = 0
        for m in range(label_size):
            label = label_list[m]
            loc = index.index(label)
            summary = summary + sum(indicator[loc:num_class]) * 1.0 / (num_class-loc+1)
            #summary = summary + sum(indicator[:loc+1]) * 1.0 / (loc+1)
        
        # 不考虑那些不属于任何一类的样本    
        if label_size > 0:
            aveprec = aveprec + summary / label_size
            num_valid_instance += 1

    Average_Precision = aveprec / num_valid_instance
    
    return Average_Precision

def OneVsRest_multilabel(Classifier, **kwargs):
    """ multi-label classification
    """
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier(Classifier(**kwargs)).fit(model_train_feature, model_train_label)
    
    model_test_pred = clf.predict_proba(model_test_feature)
    
    clf = OneVsRestClassifier(Classifier(**kwargs)).fit(train_feature, train_label)
    train_pred = clf.predict_proba(train_feature)
    test_pred = clf.predict_proba(test_feature)
    
    return model_test_pred, train_pred, test_pred
    
def transform(n_components, method = 'PCA'):
    """ transform the dataset using PCA or CCA
    """
    from sklearn.decomposition import PCA
    #from sklearn.cross_decomposition import CCA
    
    pca = PCA(n_components).fit(train_feature)
    
    model_train_feature = pca.transform(model_train_feature)
    model_test_feature = pca.transform(model_test_feature)
    test_feature = pca.transform(test_feature)
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    
    # load dataset
    f = open('task1-dataset/task1-dataset.pickle')
    import pickle
    # the following variables are accessed throughout this script, be careful
    model_train_feature, model_train_label, model_test_feature, model_test_label, test_feature = pickle.load(f)
    f.close()
    
    # construct the original train data
    train_feature = np.vstack((model_train_feature, model_test_feature))
    train_label = np.vstack((model_train_label, model_test_label))
    
    # apply PCA or CCA
    transform(n_components)
    
    #from sklearn.naive_bayes import MultinomialNB
    #method_name = 'MultinomialNB+OneVsRest'
    #model_test_pred, train_pred, test_pred = OneVsRest_multilabel(MultinomialNB)
    
    #from sklearn.svm import SVC
    #method_name = 'SVC+OneVsRest'
    #model_test_pred, train_pred, test_pred = OneVsRest_multilabel(SVC, C = 5, gamma = 0.05, probability = True)
    
    method_name = 'kNN+OneVsRest'
    from sklearn.neighbors import KNeighborsClassifier
    model_test_pred, train_pred, test_pred = OneVsRest_multilabel(KNeighborsClassifier, k=1)
    
    model_test_AP = average_precision(model_test_pred, model_test_label)
    print 'Model testing AP:', model_test_AP
    train_AP = average_precision(train_pred, train_label)
    print 'Model training AP:', train_AP
    
    # save the final prediction
    index = 0
    f = open('results/task1-%s-%d-%f-.csv' % (method_name, n_components, model_test_AP), 'w')
    n = len(test_pred)
    for i in range(n):
        result = ','.join(str(x) for x in test_pred[i, :])
        f.write('%d,%s\n' % (index+1, result))
        index += 1
        
    f.close()
