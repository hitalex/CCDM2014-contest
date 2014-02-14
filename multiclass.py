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
    from sklearn.naive_bayes import GaussianNB as NBClasifier
    
    nb_classifier = NBClasifier().fit(model_train_data, label_train)
    train_pred = nb_classifier.predict(model_train_data)
    test_pred = nb_classifier.predict(model_test_data)
    
    nb_classifier = NBClasifier().fit(train_data, label)
    y_pred = nb_classifier.predict(test_data)
    
    return train_pred, test_pred, y_pred
    
def LDA(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ Linear Discriminant Analysis, a.k.a LDF
    """
    num_class = 3
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
    
    from LDF import build_LDF_model, LDF_predict # my version of LDF
    inverse_cov, weight, w0 = build_LDF_model(num_class, model_train_data, label_train)
    
    train_pred = LDF_predict(model_train_data, num_class, inverse_cov, weight, w0)
    test_pred = LDF_predict(model_test_data, num_class, inverse_cov, weight, w0)
    
    inverse_cov, weight, w0 = build_LDF_model(num_class, train_data, label)
    y_pred = LDF_predict(test_data, num_class, inverse_cov, weight, w0)
    
    return train_pred, test_pred, y_pred
    
def QDA(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ QDF
    """
    num_class = 3
    from QDF import build_QDF_model, QDF_predict # my version of QDF
    prior, mean, cov_matrix = build_QDF_model(num_class, model_train_data, label_train)
    
    train_pred = QDF_predict(model_train_data, num_class, prior, mean, cov_matrix)
    test_pred = QDF_predict(model_test_data, num_class, prior, mean, cov_matrix)
    
    prior, mean, cov_matrix = build_QDF_model(num_class, train_data, label)
    y_pred = QDF_predict(train_data, num_class, prior, mean, cov_matrix)
    
    return train_pred, test_pred, y_pred
    
def decision_tree_classifier(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    """ DT
    """
    print 'DT classifier...'
    from sklearn import tree
    clf = tree.DecisionTreeClassifier().fit(model_train_data, label_train)
    
    train_pred = clf.predict(model_train_data)
    test_pred = clf.predict(model_test_data)
    
    clf = tree.DecisionTreeClassifier().fit(train_data, label)
    y_pred = clf.predict(test_data)
    
    return train_pred, test_pred, y_pred
    
def kNN(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data):
    print ' K nearest neighbour classification...'
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5).fit(model_train_data, label_train)
    
    train_pred = neigh.predict(model_train_data)
    test_pred = neigh.predict(model_test_data)
    
    neigh = KNeighborsClassifier(n_neighbors=5).fit(train_data, label)
    y_pred = neigh.predict(test_data)
    
    return train_pred, test_pred, y_pred

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
    # train_pred, test_pred, y_pred are prediction results of model_train_data, model_test_data, test_data
    #train_pred, test_pred, y_pred = Naive_bayes(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #train_pred, test_pred, y_pred = LDA(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    #train_pred, test_pred, y_pred = QDA(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    #train_pred, test_pred, y_pred = decision_tree_classifier(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    train_pred, test_pred, y_pred = kNN(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data)
    
    print 'Model training acc:'
    print classification_report(label_train, train_pred)    
    print 'Model testing acc:'
    print classification_report(label_test, test_pred)
    f1_score_list = f1_score(label_test, test_pred, average=None)
    print 'F1 score:', f1_score_list, 'Avg:', sum(f1_score_list) / len(f1_score_list)
    
    # save the final prediction
    index = 0
    f = open('output.csv', 'w')
    for y in y_pred:
        f.write('%d,%d\n' % (index+1, y_pred[index]))
        index += 1
        
    f.close()
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    main(n_components)
