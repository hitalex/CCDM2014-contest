#coding=utf8

"""
Gaussian Classifiers

QDF: Quadratic discriminant function
"""

import ipdb
import pprint

import numpy as np
import math
import sklearn.metrics

class QDF(object):
    def __init__(self):
        self.prior = None
        self.mean = None
        self.cov_matrix = None
        self.num_class = 0
        
    @staticmethod
    def transform_label(label_list):
        """ 将多类的类序号转换为以0开始的标签
        """
        class_labels = np.unique(label_list)
        num_class = len(class_labels)
        
        index = 0
        y = np.array(label_list, int)
        # map from index ==> original class labels
        label_map = np.array([0] * num_class)
        for c in class_labels:
            y[label_list == c] = index
            label_map[index] = c
            index += 1
            
        return num_class, y, label_map
        
    @staticmethod
    def map_class_index(pred, label_map):
        """ 将程序内部的预测结果转换成原始的结论
        """
        pred_new = np.array(pred, int)
        n = len(pred)
        for i in range(n):
            pred_new[i] = label_map[pred[i]]
            
        return pred_new
    
    def print_cov_matrix(self):
        """ Print each covariance matrix for each class
        """
        print 'Covariance matrix for each class:'
        for i in range(self.num_class):
            print 'Class %d:' % i
            pprint.pprint(self.cov_matrix[i])
            
    def fit(self, x_train, y_train):
        """ Cacualte prior prob., means and covariance matrix for each class
        """
        # infer the number of class
        self.num_class, y_train, self.label_map = QDF.transform_label(y_train)
        
        data = []
        train_count = len(x_train)
        for i in range(self.num_class):
            data.append(list())
            
        # Note: class indexes must be 0,1,2,... staring with 0
        for i in range(train_count):
            class_index = int(y_train[i])
            data[class_index].append(x_train[i])
            
        self.mean = []
        self.cov_matrix = []
        self.prior = []
        #ipdb.set_trace()
        for i in range(self.num_class):
            #data[i] = np.matrix(data[i], dtype=np.float64)
            data[i] = np.concatenate(data[i], axis=0)
            self.mean.append(data[i].mean(0).T)
            # np.cov treat each row as one feature, so data[i].T has to be transposed
            self.cov_matrix.append(np.matrix(np.cov(data[i].T)))
            self.prior.append(len(data[i]) * 1.0 / train_count)
            
        return self
        
    def predict(self, x_test):
        """ Predict class labels
        Find the class lable that maximize the prob
        """
        inverse_cov = []
        log_det_cov = []
        for i in range(self.num_class):
            det = np.linalg.det(self.cov_matrix[i])
            if det == 0:
                d, m = self.cov_matrix[i].shape
                gamma = 0.5
                cov = self.cov_matrix[i] + gamma * np.eye(d) # add a regularizer
                inverse_cov.append(cov.getI())
                det = np.linalg.det(cov)
            else:
                inverse_cov.append(self.cov_matrix[i].getI())
                
            log_det_cov.append(math.log(det))
            
        predicted_labels = []
        for row in x_test:
            x = np.matrix(row, np.float64).T
            max_posteriori = -float('inf')
            prediction = -1
            for i in range(self.num_class):
                diff = x - self.mean[i]
                p = 2 * math.log(self.prior[i]) # we do not ignore priors here
                p = p - (diff.T * inverse_cov[i] * diff)[0,0] - log_det_cov[i]
                if p > max_posteriori:
                    max_posteriori = p
                    prediction = i
                    
            predicted_labels.append(prediction)
        
        return QDF.map_class_index(predicted_labels, self.label_map)

if __name__ == "__main__":
    import sys
    
    print 'exit'
