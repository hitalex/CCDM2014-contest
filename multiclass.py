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

from dimension_reduction import PCA_transform, CCA_transform, ISOMAP_transform
from evaluation import predition2dict
from evaluation import score_list2 as f1_score_dict
from preprocessing import over_sampling

def get_classifier_by_name(method_name, paras):
    """ Get classifiers and kwargs
    """
    Classifier = None
    kwargs = {}
    
    if method_name == 'NB':
        from sklearn.naive_bayes import GaussianNB as Classifier
    elif method_name == 'SVM':
        from sklearn.svm import LinearSVC as Classifier
        kwargs = {'C': float(paras[0]), 'random_state':int(paras[1])}
    elif method_name == 'LDF':
        from LDF import LDF as Classifier
    elif method_name == 'QDF':
        from Gaussian_classifier import QDF as Classifier
    elif method_name == 'RDA':
        from Gaussian_classifier import RDA as Classifier
        kwargs = {'beta': float(paras[0]), 'gamma':float(paras[1])}
    elif method_name == 'MQDF':
        from Gaussian_classifier import MQDF as Classifier
        kwargs = {'k': int(paras[0]), 'delta0':float(paras[1])}
    elif method_name == 'DT':
        from sklearn.tree import DecisionTreeClassifier as Classifier
        kwargs = {'max_depth':20, 'min_samples_split':30, 'random_state':0}
    elif method_name == 'kNN':
        from sklearn.neighbors import KNeighborsClassifier as Classifier
        kwargs = {'k':1}
    elif method_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier as Classifier
        kwargs = {'n_estimators':50, 'max_depth':10}
    elif method_name == 'ETC':
        from sklearn.ensemble import ExtraTreesClassifier as Classifier
        kwargs = {'n_estimators':50, 'max_depth':50}
    elif method_name == 'GBT':
        from sklearn.ensemble import GradientBoostingClassifier as Classifier
    elif method_name == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier as Classifier
        kwargs = {'n_estimators':50}
    else:
        print 'Unsupported method!', method_name
        exit
        
    return Classifier, kwargs
    
def get_classifier_by_type(clftype, model_train_feature, model_train_label, Classifier, kwargs):
    """ Get classifiers
    """
    print 'Train multi-class classifiers, type = %s' % (clftype)
    if clftype == 'multiclass':
        clf = Classifier(**kwargs).fit(model_train_feature, model_train_label)
    elif clftype == 'onevsrest':
        from sklearn.multiclass import OneVsRestClassifier
        clf = OneVsRestClassifier(Classifier(**kwargs)).fit(model_train_feature, model_train_label)
    elif clftype == 'onevsone':
        from sklearn.multiclass import OneVsOneClassifier
        clf = OneVsOneClassifier(Classifier(**kwargs)).fit(model_train_feature, model_train_label)
    elif clftype == 'occ':
        from sklearn.multiclass import OutputCodeClassifier
        clf = OutputCodeClassifier(Classifier(**kwargs), code_size=2, random_state=0).fit(model_train_feature, model_train_label)
    else:
        print 'Unsupported clf type:', clftype
        sys.exit(1)
        
    return clf
    
def smote_sampling(x_train, y_train):
    """ Using SMOTE to over sampling
    Note: when sampling, x_train and x_test are both used
    """
    from SMOTE import SMOTE
    num_class = len(np.unique(y_train))
    class_count = [0] * num_class
    for i in range(num_class):
        class_count[i] = len(y_train[y_train == i])
        
    k = 5 # parameter for SMOTE algorithm
    max_class_count = max(class_count)
    for i in range(num_class):
        if class_count[i] >= max_class_count:
            continue
        N = max_class_count * 1.0 / class_count[i] * 100
        synthetic_samples = SMOTE(x_train[y_train == i, :], N, k)
        labels = np.array([i] * len(synthetic_samples), int)
        if i == 0:
            new_X = synthetic_samples
            new_Y = labels
        else:
            new_X = np.vstack((new_X, synthetic_samples))
            new_Y = np.concatenate((new_Y, labels))
            
    x_train = np.vstack((x_train, new_X))
    y_train = np.concatenate((y_train, new_Y))
    
    return x_train, y_train
    
def multiclass(train_feature, train_label, test_feature, clftype, method_name, paras):
    """ The multi classifier method
    clftype: 'multiclass', 'onevsrest', 'onevsone'
    method_name: the classifier name
    paras: list form of parameters
    """
    Classifier, kwargs = get_classifier_by_name(method_name, paras)
    
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
        
        print 'SMOTE over sampling...'
        model_train_feature, model_train_label = smote_sampling(model_train_feature, model_train_label)
        
        clf = get_classifier_by_type(clftype, model_train_feature, model_train_label, Classifier, kwargs)
        model_test_pred = clf.predict(model_test_feature)
        
        print 'Model testing acc:'
        print classification_report(model_test_label, model_test_pred)
        
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
    
    print 'SMOTE over sampling...'
    train_feature, train_label = smote_sampling(train_feature, train_label)
    print 'Train the whole multi-class classifiers...'
    clf = get_classifier_by_type(clftype, train_feature, train_label, Classifier, kwargs)
    train_pred = clf.predict(train_feature)
    test_pred = clf.predict(test_feature)
    
    print 'Model train acc:'
    print classification_report(train_label, train_pred)
    
    #f1_score_list = f1_score(train_label, train_pred, average=None)
    #avg_f1_score = sum(f1_score_list) / len(f1_score_list)
    #print 'F1 score:', f1_score_list, 'Avg:', avg_f1_score
    
    # training F1 score
    print 'Training avg F1 score:', f1_score_dict(predition2dict(train_pred), predition2dict(train_label))
    
    return method_name, test_pred, avg_avg_f1_score
    
def main(n_components, n_folds, clftype, method_name, paras):
    print 'Load dataset...'
    import pickle
    f = open('task2-dataset/task2-dataset.pickle', 'r')
    train_feature, train_label, test_feature = pickle.load(f)
    f.close()
    
    # apply the PCA dimension reduction
    #train_feature, test_feature = PCA_transform(train_feature, test_feature, 'task2-dataset/task2-PCA-decomp.mat')
    #train_feature = train_feature[:, :n_components]
    #test_feature = test_feature[:, :n_components]
    
    #print 'Apply the CCA...'
    #train_feature, test_feature = CCA_transform(train_feature, train_label, test_feature, n_components)
    
    print 'Apply the ISOMAP dimension reduction...'
    train_feature, test_feature = ISOMAP_transform(train_feature, test_feature, n_components, 5)
    #ipdb.set_trace()
    
    print 'Classifier type: %s, method: %s, paras = %r' % (clftype, method_name, paras)
    method_name, test_pred, avg_avg_f1_score = multiclass(train_feature, train_label, test_feature, clftype, method_name, paras)
    
    # save the final prediction
    """
    index = 0
    f = open('results/task2-%s-%s-%d-%f-.csv' % (clftype, method_name, n_components, avg_avg_f1_score), 'w')
    for y in test_pred:
        f.write('%d,%d\n' % (index+1, test_pred[index]))
        index += 1
        
    f.close()
    """
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    n_folds = int(sys.argv[2])
    clftype = sys.argv[3]
    method_name = sys.argv[4]
    if len(sys.argv) >= 6:
        paras = sys.argv[5:]
    else:
        paras = []
    main(n_components, n_folds, clftype, method_name, paras)
