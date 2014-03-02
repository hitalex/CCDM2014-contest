#coding=utf8

import numpy as np

def over_sampling(feature_data, label_data):
    """ Over sampling
    Method: 将所有的minority类过采样，最终使得所有类样本相近
    """
    #import ipdb; ipdb.set_trace()
    total = len(label_data)
    num_class = len(np.unique(label_data))
    class_sample_count = np.array([0] * num_class, int)
    
    for i in range(total):
        label = label_data[i]
        class_sample_count[label] += 1
    class_max_count = max(class_sample_count)
    
    index_list = []
    for i in range(num_class):
        index = np.array(range(total))[label_data == i]
        new_index = np.random.choice(index, class_max_count)
        index_list.extend(new_index)
    
    label_data.shape = (len(label_data), 1)
    data = np.hstack((feature_data, label_data))
    np.random.shuffle(index_list)
    new_data = data[index_list]
    
    new_feature_data = data[:, :-1]
    new_label = data[:, -1]
    
    new_label2 = [0] * len(new_label)
    for i in range(total):
        new_label2[i] = new_label[i, 0]
        
    new_label2 = np.array(new_label2, int)
    #new_label.shape = (len(new_label), )
    return new_feature_data, new_label2
