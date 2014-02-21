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
    
    
def multiclass(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data, Classifier, **kwargs):
    """ 
    Classifier: the classifier class
    **kwagrs: additional model parameters of the classifier
    """
    print 'Classifier name:', Classifier.__name__
    
    clf = Classifier(**kwargs).fit(model_train_data, label_train)
    model_test_pred = clf.predict(model_test_data)
    
    clf = Classifier(**kwargs).fit(train_data, label)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    
    return model_test_pred, train_pred, test_pred

    
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
    
def main(n_components):
    print 'Load dataset...'
    import pickle
    f = open('dataset/task2-PCA.pickle', 'r')
    model_train_data, label_train, model_test_data, label_test, test_data = pickle.load(f)
    f.close()
    
    # apply the PCA dimension reduction
    model_train_data = model_train_data[:, :n_components]
    model_test_data = model_test_data[:, :n_components]
    test_data = test_data[:, :n_components]
    # construct whole dataset
    train_data = np.vstack((model_train_data, model_test_data))
    label = np.hstack((label_train, label_test))
    
    # model_train_pred, model_test_pred, test_pred are prediction results of model_train_data, model_test_data, test_data
    #method_name = 'NB'
    #from sklearn.naive_bayes import GaussianNB
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, GaussianNB)
    
    #method_name = 'LDF'
    #from LDF import LDF
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, LDF)
    
    #method_name = 'QDF'
    #from QDF import QDF
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, QDF)
    
    #method_name = 'DT'
    #from sklearn import tree
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, tree.DecisionTreeClassifier, max_depth=5, min_samples_split=5, random_state=0)
    
    #method_name = 'kNN'
    #from sklearn.neighbors import KNeighborsClassifier
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, 
    #    train_data, label, test_data, KNeighborsClassifier, k=1)
    
    #method_name = 'RF'
    #from sklearn.ensemble import RandomForestClassifier
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, RandomForestClassifier, n_estimators=50, max_depth=50)
    
    #method_name = 'ETC'
    #from sklearn.ensemble import ExtraTreesClassifier
    #model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, ExtraTreesClassifier, n_estimators=50, max_depth=30)
    
    method_name = 'GBT'
    from sklearn.ensemble import GradientBoostingClassifier
    model_test_pred, train_pred, test_pred = multiclass(model_train_data, label_train, model_test_data, label_test, \
        train_data, label, test_data, GradientBoostingClassifier)
    
    #method_name = 'OneVsAllSVC'
    #from sklearn.svm import LinearSVC
    #model_test_pred, train_pred, test_pred = OneVSRest(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, LinearSVC, C=10, random_state=0)
    
    #method_name = 'OneVsAll+NB'
    #from sklearn.naive_bayes import GaussianNB
    #model_test_pred, train_pred, test_pred = OneVSRest(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, GaussianNB)
    
    print 'Model testing acc:'
    print classification_report(label_test, model_test_pred)
    f1_score_list = f1_score(label_test, model_test_pred, average=None)
    avg_f1_score = sum(f1_score_list) / len(f1_score_list)
    print 'F1 score:', f1_score_list, 'Avg:', avg_f1_score
    
    print 'Model training acc:'
    print classification_report(label, train_pred)
    
    # save the final prediction
    index = 0
    f = open('results/task2-%s-%d-%f-.csv' % (method_name, n_components, avg_f1_score), 'w')
    for y in test_pred:
        f.write('%d,%d\n' % (index+1, test_pred[index]))
        index += 1
        
    f.close()
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    main(n_components)
