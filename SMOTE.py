# coding=utf8

"""
该脚本实现以下算法：
1，采用SMOTE算法对minority samples进行over-sampling
2，（可选）对majority samples进行down-sampling
3，建立分类器模型
"""
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
    
def SMOTE(minority_samples, N, k):
    """
    The SMOTE algorithm, please refer to: [JAIR'02]SMOTE - Synthetic Minority Over-sampling Technique
    minority_samples The minority sample array
    N Amount of SMOTE N%
    k Number of nearest neighbors
    
    @return (N/100)*len(minority_samples) synthetic minority class samples
    """
    T = len(minority_samples) # number of minority samples
    if N < 100:
        T = N * 1.0 / 100 * T
        N = 100
    N = int(N * 1.0 / 100)
    
    neigh = NearestNeighbors(n_neighbors = k, radius=1.0, algorithm='auto', leaf_size=30, p=2)
    neigh = neigh.fit(minority_samples)
    
    synthetic_samples = []
    for i in range(T):
        target_sample = minority_samples[i]
        tmp = neigh.kneighbors(target_sample, k, return_distance=False)
        nnarray = tmp[0]
        populate(minority_samples, N, k, i, nnarray, synthetic_samples)
        
    return np.array(synthetic_samples, float)
        
def populate(minority_samples, N, k, i, nnarray, synthetic_samples):
    """
    根据target_sample扩充整个sample集合，并存放如synthetic_samples中
    """
    target_sample = minority_samples[i]
    numattrs = len(target_sample) # number of attr
    while N > 0:
        nn = random.choice(range(k)) # make suer nn >=1 and nn <= k
        dif = [0] * numattrs
        tmp = [0] * numattrs
        for attr in range(numattrs):
            dif = minority_samples[nnarray[nn]][attr] - target_sample[attr]
            gap = random.random() # gap >=0 and gap < 1
            tmp[attr] = target_sample[attr] + gap * dif
            
        synthetic_samples.append(tmp)
        N -= 1

