#coding=utf8

"""
相对与第1类，第0类和第2类样本数量较少，几乎是两者之和，所以这里采用两步走的策略：
1，将第0类和第2类看作第0类，和第1类组成两类问题，训练模型1
2，第0类和第2类组成两类问题，得到模型2
所以，此方法的目标就是构造两个二分类模型
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def twostep_predict(clf1, clf2, test_data):
    """ 分两步预测
     model1_label, model2_label: 分别是clf1和clf2两个模型关于
    """
    model1_pred = clf1.predict(test_data)
    model2_data = test_data[model1_pred == 0]
    model2_pred = clf2.predict(model2_data)
    
    n = len(model1_pred)
    pred = np.array([0] * n, int)
    j = 0
    for i in range(n):
        if model1_pred[i] == 1:
            pred[i] = 1
        else:
            pred[i] = model2_pred[j]
            j += 1
            
    return pred 
   
def make_twostep_dataset(train_data, label):
    #import ipdb; ipdb.set_trace()
    # 构造模型1的label：将第0类和第2类统一归为第0类
    model1_data = train_data
    model1_label = np.array(label, int)
    model1_label[model1_label == 2] = 0
    
    # 构造模型2的label：类别标签分别是0和2
    model2_label = np.array(label, int)
    model2_data = train_data[model2_label != 1, :]
    model2_label = model2_label[model2_label != 1]
    
    return model1_data, model1_label, model2_data, model2_label

def twostep(model_train_data, label_train, model_test_data, label_test, train_data, label, test_data, Classifier, **kwargs):
    """
    Classifier: the classifier class
    **kwd: additional model parameters for the classifier
    Return:
    model_test_pred: 在model_test_data上的准确率，用于模型选择
    train_pred: 在整个train_data上的准确率，用于查看是否发生过拟合
    test_pred: test_data上的准确率，是最终结果
    """    
    #import ipdb; ipdb.set_trace()
    # twostep method
    model1_data, model1_label, model2_data, model2_label = make_twostep_dataset(model_train_data, label_train)
    # 构造两个模型
    clf1 = Classifier(**kwargs).fit(model1_data, model1_label)
    clf2 = Classifier(**kwargs).fit(model2_data, model2_label)

    model_test_pred = twostep_predict(clf1, clf2, model_test_data)
    
    model1_data, model1_label, model2_data, model2_label = make_twostep_dataset(train_data, label)
    # 构造两个模型
    clf1 = Classifier(**kwargs).fit(model1_data, model1_label)
    clf2 = Classifier(**kwargs).fit(model2_data, model2_label)
    # 测试两步模型中每步的准确率
    print 'Step 1 acc:'
    y_pred = clf1.predict(model1_data)
    print classification_report(model1_label, y_pred)
    print 'Step 2 acc:'
    y_pred = clf2.predict(model2_data)
    print classification_report(model2_label, y_pred)
    
    train_pred = twostep_predict(clf1, clf2, train_data)
    
    
    test_pred = twostep_predict(clf1, clf2, test_data)
    
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
    
    #from sklearn.naive_bayes import GaussianNB
    #model_test_pred, train_pred, test_pred = twostep(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, GaussianNB)
        
    from sklearn.svm import LinearSVC
    model_test_pred, train_pred, test_pred = twostep(model_train_data, label_train, model_test_data, label_test, \
        train_data, label, test_data, LinearSVC, random_state=0, C = 10)
    
    #from QDF import QDF
    #model_test_pred, train_pred, test_pred = twostep(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, QDF)
    
    #from LDF import LDF
    #model_test_pred, train_pred, test_pred = twostep(model_train_data, label_train, model_test_data, label_test, \
    #    train_data, label, test_data, LDF)
    
    
    print 'Training acc for checking overfitting:'
    print classification_report(label, train_pred)
    
    print 'Model testing acc for model selection:'
    print classification_report(label_test, model_test_pred)
    f1_score_list = f1_score(label_test, model_test_pred, average=None)
    print 'F1 score:', f1_score_list, 'Avg:', sum(f1_score_list) / len(f1_score_list)
    
    # save the final prediction
    index = 0
    f = open('twostep_output.csv', 'w')
    for y in test_pred:
        f.write('%d,%d\n' % (index+1, test_pred[index]))
        index += 1
        
    f.close()
    
if __name__ == '__main__':
    import sys
    n_components = int(sys.argv[1])
    main(n_components)
