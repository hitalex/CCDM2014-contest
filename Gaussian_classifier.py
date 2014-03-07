#coding=utf8

"""
Gaussian Classifiers

QDF: Quadratic discriminant function
RDA: Regularized discriminant analysis
"""

import ipdb
import pprint

import numpy as np
from numpy import linalg
import numpy.matlib
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
        #import ipdb; ipdb.set_trace()
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
        Note: x_train must be 2d array, y_train must be 1d array
        """
        #import ipdb; ipdb.set_trace()
        # infer the number of class
        self.num_class, y_train, self.label_map = QDF.transform_label(y_train)
        
        data = []
        train_count = len(x_train)
        for i in range(self.num_class):
            data.append(list())
            
        # Note: class indexes must be 0,1,2,... staring with 0
        for i in range(train_count):
            class_index = int(y_train[i])
            data[class_index].append([x_train[i, :]])
            
        self.mean = []
        self.cov_matrix = []
        self.prior = []
        #ipdb.set_trace()
        for i in range(self.num_class):
            #data[i] = np.matrix(data[i], dtype=np.float64)
            data[i] = np.matrix(np.concatenate(data[i], axis=0), dtype=np.float64)
            self.mean.append(data[i].mean(0).T)
            # np.cov treat each row as one feature, so data[i].T has to be transposed
            self.cov_matrix.append(np.matrix(np.cov(data[i].T)))
            
            #self.prior.append(len(data[i]) * 1.0 / train_count)
            self.prior.append(1)
            
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
        
class RDA(QDF):
    def __init__(self, beta = 0, gamma = 0):
        """ beta and gama are hyper-parameters
        """
        super(RDA, self).__init__()
        self.beta = beta
        self.gamma = gamma
        
    def fit(self, x_train, y_train):
        #import ipdb; ipdb.set_trace()
        super(RDA, self).fit(x_train, y_train)
        beta = self.beta
        gamma = self.gamma
        # cacualte the shared covariance matirx
        avg_cov = np.matlib.zeros(self.cov_matrix[0].shape)
        for i in range(self.num_class):
            avg_cov += (self.prior[i] * self.cov_matrix[i])
            
        num_feature = len(x_train[0])
        for i in range(self.num_class):
            # the following formula is from PPT
            tmp = self.cov_matrix[i]
            dev = tmp.trace()[0,0] * 1.0 / num_feature
            self.cov_matrix[i] = (1-gamma) * ((1-beta)*tmp + beta*avg_cov) + gamma * dev * np.matlib.eye(num_feature)
            """
            # refer to Statistical Pattern Recognition, page 43
            tmp = self.cov_matrix[i]
            tmp = (1-beta) * prior[i] * tmp + beta * avg_cov
            dev = tmp.trace()[0,0] * 1.0 / num_feature
            self.cov_matrix[i] = (1-gamma) * tmp + gamma * dev * np.matlib.eye(num_feature)
            """
        return self
        
class MQDF(QDF):
    def __init__(self, k, delta0 = None):
        """
        @k and @delta are hyper-parameters
        Note: There are three possible ways to set @delta:
        1) @delta is a hyper-parameter, set by cross validation
        2) @delta can be estimated via ML estimation, in which case, this delta 
            is no longer a hyper-parameter
        3) @delta can be set close to $\sigma_{i,k}$ or $\sigma_{i,k+1}$, in which case, 
            this delta is also not a hyper-parameter
        """
        super(MQDF, self).__init__()
        self.k = k
        self.delta0 = delta0
        
    def fit(self, x_train, y_train):
        super(MQDF, self).fit(x_train, y_train)
        # k, delta0, and num_class will not be modified during fit
        k = self.k
        delta0 = self.delta0
        train_count, self.num_feature = x_train.shape
        d = self.num_feature
        num_class = self.num_class
        assert(k<d and k>0)
        
        self.eigenvalue = []    # store the first largest k eigenvalues of each class
        self.eigenvector = []   # the first largest k eigenvectors, column-wise of each class
        self.delta = [0] * num_class # deltas for each class
        for i in range(num_class):
            cov = self.cov_matrix[i]
            eig_values, eig_vectors = linalg.eigh(cov)
            # sort the eigvalues
            idx = eig_values.argsort()
            idx = idx[::-1] # reverse the array
            eig_values = eig_values[idx]
            eig_vectors = eig_vectors[:,idx]
            
            self.eigenvector.append(eig_vectors[:, :k])
            self.eigenvalue.append(eig_values[:k])
            
            # delta via ML estimation
            #self.delta[i] = (cov.trace() - sum(self.eigenvalue[i])) * 1.0 / (d-k)
            
            # delta close to $\sigma_{i,k-1}$ or $\sigma_{i,k}$
            #self.delta[i] = (eig_values[k-1] + eig_values[k])/2
            #print 'Suggestd delta[%d]: %f' % (i, self.delta[i])
            
            # delta as the mean of minor values
            #print 'The minor eigen values of class %d: %r' % (i, eig_values[k:])
            #self.delta[i] = sum(eig_values[k:]) / len(eig_values[k:])
            
            self.delta[i] = self.delta0
        
        return self
        
    def predict(self, x_test):
        d = self.num_feature
        num_class = self.num_class
        k = self.k
        
        y_pred = np.array([0] * len(x_test), int)
        index = 0
        for row in x_test:
            x = np.matrix(row, np.float64).T
            max_posteriori = -float('inf')
            prediction = -1
            for i in range(num_class):
                dis = np.linalg.norm(x.reshape((d,)) - self.mean[i].reshape((d,))) ** 2 # 2-norm
                # Mahalanobis distance
                #import ipdb; ipdb.set_trace()
                ma_dis = [0] * k
                for j in range(k):
                    ma_dis[j] = (((x - self.mean[i]).T * self.eigenvector[i][:, j])[0,0])**2
                
                p = 0
                for j in range(k):
                    p += (ma_dis[j] * 1.0 / self.eigenvalue[i][j])
                
                p += ((dis - sum(ma_dis)) / self.delta[i])
                
                for j in range(k):
                    p += math.log(self.eigenvalue[i][j])
                    
                p += ((d-k) * math.log(self.delta[i]))
                p = -p
                    
                if p > max_posteriori:
                    max_posteriori = p
                    prediction = i
                    
            y_pred[index] = prediction
            index += 1
            
        return y_pred

if __name__ == "__main__":
    import sys
    
    print 'exit'
