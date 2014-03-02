#coding=utf8

"""
Calibrated label ranking: a label ranking method

Ref: A Review On Multi-Label Learning Algorithms, P17

注意：
1，对于那些训练集中不属于任何标签的数据，此方法并不包含这些
"""

import numpy as np

from dimension_reduction import PCA
from multilabel import average_precision, save_multilabel_result

def calibrated_label_ranking_predict(test_feature, num_class, classifier_matrix):
    """ The calibrated label ranking method: test process
    """
    test_count, num_feature = test_feature.shape
    pred = np.zeros((test_count, num_class), float)
    for i in range(test_count):
        ins = test_feature[i, :]
        votes = np.array([0] * num_class, float)
        for j in range(num_class):
            for k in range(0, j):
                clf = classifier_matrix[k, j]
                if clf.predict(ins) < 0:
                    votes[j] += 1
                    
            for k in range(j+1, num_class):
                clf = classifier_matrix[j, k]
                if clf.predict(ins) > 0:
                    votes[j] += 1
                    
        assert(sum(votes) > 0)
        pred[i, :] = votes * 1.0 / sum(votes)
        
    return pred

def calibrated_label_ranking_train(train_feature, train_label, BinaryClassifier, kwdargs):
    """ The calibrated label ranking method: training process
    BinaryClassifier: the base binary classifier
    **kwdargs: additional args for the binary classifier
    """
    train_count, num_feature = train_feature.shape
    train_count, num_class = train_label.shape
    
    # dataset_matrix[j, k] : the dataset index list for label pair (j,k), where j < k
    # which means more than half of the matrix is filled with None type
    dataset_matrix = np.ndarray(shape=(num_class, num_class), dtype=object)
    # init the dataset_matrix
    for j in range(0, num_class):
        for k in range(j+1, num_class):
            dataset_matrix[j, k] = list()
        
    for i in range(train_count):
        labels_info = train_label[i, :]
        relevant_labels = np.array(range(num_class))[labels_info == 1]
        # ignore samples which has no relevant lables or has no irrelevant lables
        if len(relevant_labels) == 0 or len(relevant_labels) == num_class:
            continue
            
        not_relevant_labels = list(set(range(num_class)) - set(relevant_labels))
        # add sample to dataset with label
        for rlabel in relevant_labels:
            for nrlabel in not_relevant_labels:
                if rlabel < nrlabel:
                    j = rlabel
                    k = nrlabel
                    label = +1 # sample label for this sample in new dataset
                else:
                    j = nrlabel
                    k = rlabel
                    label = -1
                # instance with label info
                ins = np.concatenate((np.array(train_feature[i, :]).flatten(), [label]))
                dataset_matrix[j, k].append(ins)
                
    # train the q(q-1)/2 classifiers
    for j in range(0, num_class):
        for k in range(j+1, num_class):
            dataset_matrix[j, k] = np.array(dataset_matrix[j, k])
            b_train_feature = dataset_matrix[j, k][:, :-1]
            b_train_label = dataset_matrix[j, k][:, -1]
            clf = BinaryClassifier(**kwdargs).fit(b_train_feature, b_train_label)
            # reuse the dataset_matrix as classifier_matrix
            dataset_matrix[j, k] = clf
            
    return dataset_matrix

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
    train_feature, test_feature = PCA(train_feature, test_feature, 'task1-dataset/task1-PCA-decomp.mat')
    train_feature = train_feature[:, :n_components]
    test_feature = test_feature[:, :n_components]
    
    train_count, num_class = train_label.shape
    
    from sklearn.naive_bayes import GaussianNB as BinaryClassifier
    method_name = 'CLR+NB'
    kwdargs = {}
    
    #from sklearn.svm import SVC as BinaryClassifier
    #method_name = 'CLR+SVM'
    #kwdargs = {'C':5, 'gamma':0.05}
    
    #from sklearn.neighbors import KNeighborsClassifier as BinaryClassifier
    #method_name = 'CLR+kNN'
    #kwdargs = {'k':1}
    
    #method_name = 'CLR+DT'
    #from sklearn.tree import DecisionTreeClassifier as BinaryClassifier
    #kwdargs = {'max_depth':20, 'min_samples_split':2}
    
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
        classifier_matrix = calibrated_label_ranking_train(model_train_feature, model_train_label, BinaryClassifier, kwdargs)
        model_test_pred = calibrated_label_ranking_predict(model_test_feature, num_class, classifier_matrix)
        
        #import ipdb; ipdb.set_trace()
        model_test_AP[index] = average_precision(model_test_pred, model_test_label)
        print 'Model testing AP:', model_test_AP[index]
        
        index += 1
    
    avg_AP = sum(model_test_AP) / n_folds
    print '\nAverage AP:', avg_AP
    
    print 'Train the whole dataset...'
    classifier_matrix = calibrated_label_ranking_train(train_feature, train_label, BinaryClassifier, kwdargs)
    print 'Predict...'
    train_pred = calibrated_label_ranking_predict(train_feature, num_class, classifier_matrix)
    test_pred = calibrated_label_ranking_predict(test_feature, num_class, classifier_matrix)
    save_multilabel_result(method_name, n_components, avg_AP, test_pred)
