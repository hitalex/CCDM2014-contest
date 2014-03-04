#coding=utf8

"""
Gaussian Classifiers

LDF: Linear discriminant function
Case: All classes share the same covariance matrix
"""
import math
import pdb
import sys

import numpy as np
import numpy.matlib
import sklearn.metrics

from QDF import QDF

class LDF(object):
    def __init__(self):
        self.inverse_cov = None
        self.weight = None
        self.w0 = None
        
    def fit(self, x_train, y_train):
        """ LDF model
        First call QDF model to caculate the mean, cov_matirx
        """
        #import ipdb; ipdb.set_trace()
        self.num_class, y_train, self.label_map = QDF.transform_label(y_train)
        qdf = QDF().fit(x_train, y_train)
        prior = qdf.prior
        mean = qdf.mean
        cov_matrix = qdf.cov_matrix
        #print_cov_matrix(cov_matrix)
        
        # cacualte the shared covariance matirx
        avg_cov = np.matlib.zeros(cov_matrix[0].shape)
        for i in range(self.num_class):
            avg_cov += (prior[i] * cov_matrix[i])
            
        self.inverse_cov = avg_cov.getI() # get the inverse covariance matrix
        
        num_feature = x_train.shape[1]
        # each column for weight[i]
        weight = np.matrix([0] * num_feature).T
        self.w0 = []
        for i in range(self.num_class):
            wi = 2 * self.inverse_cov.T * mean[i]
            weight = np.hstack((weight, wi))
            
            wi0 = 2 * math.log(prior[i]) - (mean[i].T * self.inverse_cov * mean[i])[0,0]
            self.w0.append(wi0)
            
        self.weight = weight[:, 1:]
        
        return self
        
    def predict(self, x_test):
        predicted_labels = []
        for row in x_test:
            x = np.matrix(row, np.float64).T
            max_posteriori = -float('inf')
            prediction = -1
            for i in range(self.num_class):
                p = (-1 * (x.T * self.inverse_cov * x) + self.weight[:, i].T * x + self.w0[i])[0,0]
                #p = (self.weight[:, i].T * x + self.w0[i])[0,0]
                if p > max_posteriori:
                    max_posteriori = p
                    prediction = i
                    
            predicted_labels.append(prediction)
            
        return QDF.map_class_index(predicted_labels, self.label_map)
    
def main(dataset_name):
    import readdata
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    inverse_cov, weight, w0 = build_LDF_model(num_class, x_train, y_train)

    y_pred = LDF_predict(x_test, num_class, inverse_cov, weight, w0)
    #pdb.set_trace()
    print sklearn.metrics.classification_report(y_test, y_pred)
    
    print 'Average accuracy: ', sklearn.metrics.accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(dataset_name)    
