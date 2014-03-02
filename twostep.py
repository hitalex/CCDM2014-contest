#coding=utf8

"""
相对与第1类，第0类和第2类样本数量较少，几乎是两者之和，所以这里采用两步走的策略：
1，将第0类和第2类看作第0类，和第1类组成两类问题，训练模型1
2，第0类和第2类组成两类问题，得到模型2
所以，此方法的目标就是构造两个二分类模型
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dimension_reduction import PCA_transform
from evaluation import predition2dict
from evaluation import score_list2 as f1_score_dict

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

def twostep(train_feature, train_label, test_feature, Classifier, kwargs):
    """
    Classifier: the classifier class
    **kwd: additional model parameters for the classifier
    Return:
    train_pred: 在整个train_data上的准确率，用于查看是否发生过拟合
    test_pred: test_data上的准确率，是最终结果
    """    
    #import ipdb; ipdb.set_trace()
    # twostep method
    model1_feature, model1_label, model2_feature, model2_label = make_twostep_dataset(train_feature, train_label)
    # 构造两个模型
    clf1 = Classifier(**kwargs).fit(model1_feature, model1_label)
    clf2 = Classifier(**kwargs).fit(model2_feature, model2_label)

    train_pred  = twostep_predict(clf1, clf2, train_feature)
    test_pred = twostep_predict(clf1, clf2, test_feature)
    
    # 测试两步模型中每步的准确率
    print 'Step 1 acc:'
    y_pred = clf1.predict(model1_feature)
    print classification_report(model1_label, y_pred)
    print 'Step 2 acc:'
    y_pred = clf2.predict(model2_feature)
    print classification_report(model2_label, y_pred)
    
    return train_pred, test_pred

def main(n_components, n_folds, method_name):
    print 'Load dataset...'
    import pickle
    f = open('task2-dataset/task2-dataset.pickle', 'r')
    train_feature, train_label, test_feature = pickle.load(f)
    f.close()
    
    train_feature, test_feature = PCA_transform(train_feature, test_feature, 'task2-dataset/task2-PCA-decomp.mat')
    train_feature = train_feature[:, :n_components]
    test_feature = test_feature[:, :n_components]
    
    kwargs = {}
    #from sklearn.naive_bayes import GaussianNB as Classifier
    #method_name = 'Twostep+NB'
        
    #from sklearn.svm import LinearSVC as Classifier
    #method_name = 'Twostep+SVC'
    #kwargs = {'random_state':0, 'C':10}
    
    from QDF import QDF as Classifier
    method_name = 'Twostep+QDF'
    
    #from LDF import LDF as Classifier
    #method_name = 'Twostep+LDF'
    
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
        
        model_train_pred, model_test_pred = twostep(model_train_feature, model_train_label, model_test_feature, Classifier, kwargs)

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
    
    print 'Train the whole multi-class classifiers...'
    train_pred, test_pred = twostep(train_feature, train_label, test_feature, Classifier, kwargs)
    # training F1 score
    print 'Training avg F1 score:', f1_score_dict(predition2dict(train_pred), predition2dict(train_label))
    
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
    n_folds = int(sys.argv[2])
    method_name = sys.argv[3]
    main(n_components, n_folds, method_name)
