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

def Naive_bayes(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    print 'Naive bayes classifier...'
    from sklearn.naive_bayes import GaussianNB as NBClasifier
    
    nb_classifier = NBClasifier().fit(model_train_data, label_train)
    model_test_pred = nb_classifier.predict(model_test_data)
    
    nb_classifier = NBClasifier().fit(train_data, label)
    train_pred = nb_classifier.predict(train_data)
    test_pred = nb_classifier.predict(test_data)
    
    return model_test_pred, train_pred, test_pred
    
def LDF(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ Linear Discriminant Analysis, a.k.a LDF
    """
    """
    from sklearn.lda import LDA
    # caculate priors
    prior = np.array([0] * num_class)
    for y in label:
        prior[y] += 1
    prior = prior * 1.0 / len(label)
    
    clf = LDA().fit(model_train_data, label_train)
    train_pred = clf.predict(model_train_data)
    test_pred = clf.predict(model_test_data)
    
    clf = LDA(priors=prior).fit(train_data, label)
    y_pred = clf.predict(test_data)
    """
    
    print 'LDF...'
    from LDF import LDF
    ldf_clf = LDF().fit(model_train_data, label_train)
    
    test_pred = ldf_clf.predict(model_test_data)
    
    ldf_clf = LDF().fit(train_data, label)
    train_pred = ldf_clf.predict(train_data)
    y_pred = ldf_clf.predict(test_data)
    
    return test_pred, train_pred, y_pred
    
def QDF(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ QDF
    """
    print 'QDF...'
    from QDF import QDF
    qdf_clf = QDF().fit(model_train_data, label_train)
    
    test_pred = qdf_clf.predict(model_test_data)
    
    ldf_clf = QDF().fit(train_data, label)
    train_pred = qdf_clf.predict(train_data)
    y_pred = qdf_clf.predict(test_data)
    
    return test_pred, train_pred, y_pred
    
def decision_tree_classifier(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ DT
    """
    print 'DT classifier...'
    from sklearn import tree
    clf = tree.DecisionTreeClassifier().fit(model_train_data, label_train)
    
    test_pred = clf.predict(model_test_data)
    
    clf = tree.DecisionTreeClassifier().fit(train_data, label)
    train_pred = ldf_clf.predict(train_data)
    y_pred = clf.predict(test_data)
    
    return train_pred, test_pred, y_pred
    
def kNN(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    print ' K nearest neighbour classification...'
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5).fit(model_train_data, label_train)
    
    model_test_pred = neigh.predict(model_test_data)
    
    neigh = KNeighborsClassifier(n_neighbors=5).fit(train_data, label)
    train_pred = neigh.predict(train_data)
    test_pred = neigh.predict(test_data)
    
    return model_test_pred, train_pred, test_pred
    
def random_forest(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    # random forest
    print 'Random forest classifier...'
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(n_estimators = 10, max_depth = 10).fit(model_train_data, label_train)
    
    model_test_pred = clf.predict(model_test_data)
    
    clf = RandomForestClassifier(n_estimators = 10, max_depth = 10).fit(train_data, label)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    
    return model_test_pred, train_pred, test_pred
    
def OneVSRest_SVC(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    # One vs rest + SVC
    print 'OneVSRest + SVC...'
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    
    clf = OneVsRestClassifier(LinearSVC(random_state=0, C = 10)).fit(model_train_data, label_train)
    model_test_pred = clf.predict(model_test_data)
    
    clf = OneVsRestClassifier(LinearSVC(random_state=0, C = 10)).fit(train_data, label)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    
    return model_test_pred, train_pred, test_pred
    
def OneVSRest_GaussianNB(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    # One vs rest + GaussianNB
    print 'One vs rest + GaussianNB'
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.naive_bayes import GaussianNB
    
    clf = OneVsRestClassifier(GaussianNB()).fit(model_train_data, label_train)
    model_test_pred = clf.predict(model_test_data)
    
    clf = OneVsRestClassifier(GaussianNB()).fit(train_data, label)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    
    return model_test_pred, train_pred, test_pred
    
def voting(pred_matrix):
    # majority vote
    #import ipdb; ipdb.set_trace()
    num_class = 3
    pred_matrix = np.array(pred_matrix)
    
    num_clf, num_sample = pred_matrix.shape
    votes = [0] * num_class
    result = []
    for i in range(num_sample):
        pred = pred_matrix[:, i]
        for j in range(num_clf):
            votes[pred[j]] += 1
        
        y = votes.index(max(votes))
        result.append(y)
        
    return result
    
def majority_vote(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ Majority vote method
    """
    print 'Majority vote...'
    # pred_matrix: n_classifier * n_samples 
    train_pred_matrix = []
    test_pred_matrix = []
    y_pred_matrix = []
    
    train_pred, test_pred, y_pred = Naive_bayes(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = LDA(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = QDA(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = decision_tree_classifier(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = kNN(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = random_forest(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = OneVSRest_SVC(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    train_pred, test_pred, y_pred = OneVSRest_GaussianNB(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    train_pred_matrix.append(train_pred)
    test_pred_matrix.append(test_pred)
    y_pred_matrix.append(y_pred)
    
    print 'Number of ensemble classifiers:', len(test_pred_matrix)
    train_vote_result = voting(train_pred_matrix)
    test_vote_result = voting(test_pred_matrix)
    y_vote_result = voting(y_pred_matrix)
    
    return train_vote_result, test_vote_result, y_vote_result
    
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
    #model_test_pred, train_pred, test_pred = Naive_bayes(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = LDF(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    model_test_pred, train_pred, test_pred = QDF(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = decision_tree_classifier(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = kNN(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = random_forest(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = OneVSRest_SVC(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = OneVSRest_GaussianNB(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #model_test_pred, train_pred, test_pred = majority_vote(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)

    print 'Model testing acc:'
    print classification_report(label_test, model_test_pred)
    f1_score_list = f1_score(label_test, model_test_pred, average=None)
    print 'F1 score:', f1_score_list, 'Avg:', sum(f1_score_list) / len(f1_score_list)
    
    print 'Model training acc:'
    print classification_report(label, train_pred)
    
    # save the final prediction
    index = 0
    f = open('output.csv', 'w')
    for y in test_pred:
        f.write('%d,%d\n' % (index+1, test_pred[index]))
        index += 1
        
    f.close()
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    main(n_components)
