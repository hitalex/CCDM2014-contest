#coding=utf8

"""
Multi-class classification using different methods
Types:
Inherently supports multi-class：NB, Decision Trees, LDA, QDA, MQDF, Neural networks, nearest neighbour, GMM
One-vs-rest
One-vs-one
Algorithms:
SVM, DT, NN, 

注意：
1，在测试模型时，使用model-test数据集，但是在训练最终模型时，需要用model-train和model-test的并集以得到更多的训练数据
"""

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report

import ipdb

from dimension_reduction import PCA_transform, CCA_transform
from evaluation import predition2dict
from evaluation import score_list2 as f1_score_dict
from preprocessing import over_sampling

def get_classifier_by_name(method_name):
    """ Get classifiers and kwdargs
    """
    Classifier = None
    kwdargs = {}
    
    if method_name == 'NB':
        from sklearn.naive_bayes import GaussianNB as Classifier
    elif method_name == 'LDF':
        from LDF import LDF as Classifier
    elif method_name == 'QDF':
        from sklearn.tree import DecisionTreeClassifier as Classifier
    elif method_name == 'DT':
        from sklearn.tree import DecisionTreeClassifier as Classifier
        kwdargs = { 'max_depth':20, 'min_samples_split':30, 'random_state':0}
    elif method_name == 'kNN':
        from sklearn.neighbors import KNeighborsClassifier as Classifier
        kwdargs = {'k':1}
    elif method_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier as Classifier
        kwdargs = {'n_estimators':50, 'max_depth':10}
    elif method_name == 'ETC':
        from sklearn.ensemble import ExtraTreesClassifier as Classifier
        kwdargs = {'n_estimators':50, 'max_depth':50}
    elif method_name == 'GBT':
        from sklearn.ensemble import GradientBoostingClassifier as Classifier
    elif method_name == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier as Classifier
        kwdargs = {'n_estimators':50}
    else:
        print 'Unsupported method!', method_name
        exit
        
    return Classifier, kwdargs
    
def multiclass(train_feature, train_label, test_feature, method_name):
    """ The multi classifier method
    """
    Classifier, kwdargs = get_classifier_by_name(method_name)
    
    #method_name = 'OneVsAllSVC'
    #from sklearn.svm import LinearSVC
    #model_test_pred, train_pred, test_pred = OneVSRest(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, LinearSVC, C=10, random_state=0)
    
    #method_name = 'OneVsAll+NB'
    #from sklearn.naive_bayes import GaussianNB
    #model_test_pred, train_pred, test_pred = OneVSRest(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, GaussianNB)
    
    print 'Method: ', method_name
    from sklearn.cross_validation import KFold
    kf = KFold(len(train_label), n_folds, indices=True)
    index = 0
    avg_f1_score_list = [0] * n_folds
    for train_index, test_index in kf:
        print 'Prepare cv dataset: %d' % index
        model_train_feature = train_feature[train_index, :]
        model_test_feature = train_feature[test_index, :]
        model_train_label = train_label[train_index]
        model_test_label = train_label[test_index]
        
        #print 'Over sampling...'
        #model_train_feature, model_train_label = over_sampling(model_train_feature, model_train_label)
        #ipdb.set_trace()
        
        print 'Train multi-class classifiers...'
        clf = Classifier(**kwdargs).fit(model_train_feature, model_train_label)
        model_test_pred = clf.predict(model_test_feature)
        
        #print 'Model testing acc:'
        #print classification_report(model_test_label, model_test_pred)
        
        #f1_score_list = f1_score(model_test_label, model_test_pred, average=None)
        #avg_f1_score_list[index] = sum(f1_score_list) / len(f1_score_list)
        #print 'F1 score:', f1_score_list, 'Avg:', avg_f1_score_list[index]
        
        avg_f1_score_list[index] = f1_score_dict(predition2dict(model_test_pred), predition2dict(model_test_label))
        print 'Avg: ', avg_f1_score_list[index]
        
        index += 1
    
    print 'Method:', method_name
    avg_avg_f1_score = sum(avg_f1_score_list) / len(avg_f1_score_list)
    print 'Avg avg_f1_score:', avg_avg_f1_score, '\n'
    
    #print 'Oversampling...'
    #train_feature, train_label = over_sampling(train_feature, train_label)
    
    print 'Train the whole multi-class classifiers...'
    clf = Classifier(**kwdargs).fit(train_feature, train_label)
    train_pred = clf.predict(train_feature)
    test_pred = clf.predict(test_feature)
    
    #print 'Model train acc:'
    #print classification_report(train_label, train_pred)
    
    #f1_score_list = f1_score(train_label, train_pred, average=None)
    #avg_f1_score = sum(f1_score_list) / len(f1_score_list)
    #print 'F1 score:', f1_score_list, 'Avg:', avg_f1_score
    
    # training F1 score
    print 'Training avg F1 score:', f1_score_dict(predition2dict(train_pred), predition2dict(train_label))
    
    return method_name, test_pred, avg_avg_f1_score
    
def OneVSRest(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data, Classifier, **kwargs):
    # One vs rest + SVC
    print 'OneVSRest + %s' % Classifier.__name__
    from sklearn.multiclass import OneVsRestClassifier
    
    clf = OneVsRestClassifier(Classifier(**kwargs)).fit(model_train_data, label_train)
    model_test_pred = clf.predict(model_test_data)
    
    clf = OneVsRestClassifier(Classifier(**kwargs)).fit(train_data, label)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    
    return model_test_pred, train_pred, test_pred
    
def main(n_components, n_folds, method_name):
    print 'Load dataset...'
    import pickle
    f = open('task2-dataset/task2-dataset.pickle', 'r')
    train_feature, train_label, test_feature = pickle.load(f)
    f.close()
    
    # apply the PCA dimension reduction
    #train_feature, test_feature = PCA_transform(train_feature, test_feature, 'task2-dataset/task2-PCA-decomp.mat')
    #train_feature = train_feature[:, :n_components]
    #test_feature = test_feature[:, :n_components]
    
    print 'Apply the CCA...'
    train_feature, test_feature = CCA_transform(train_feature, train_label, test_feature, n_components)
    
    method_name, test_pred, avg_avg_f1_score = multiclass(train_feature, train_label, test_feature, method_name)
    
    # save the final prediction
    index = 0
    f = open('results/task2-%s-%d-oversampling-%f-.csv' % (method_name, n_components, avg_avg_f1_score), 'w')
    for y in test_pred:
        f.write('%d,%d\n' % (index+1, test_pred[index]))
        index += 1
        
    f.close()
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    n_folds = int(sys.argv[2])
    method_name = sys.argv[3]
    main(n_components, n_folds, method_name)
