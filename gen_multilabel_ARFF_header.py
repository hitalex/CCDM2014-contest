#coding=utf8


def main():
    arff_file = open('/home/kqc/dataset/CCDM2014/task1/Task1-ARFF-header', 'w')
    
    num_feature = 129
    num_label = 12
    
    arff_file.write('@RELATION CCDM2014-Task1: -C -%d\n\n' % num_label)
    
    for i in range(num_feature):
        arff_file.write('@ATTRIBUTE f%d {0,1}\n' % i)
        
    arff_file.write('\n')
    for i in range(num_label):
        arff_file.write('@ATTRIBUTE class%d {-1,1}\n' % i)
        
    arff_file.write('\n@DATA\n')

if __name__ == '__main__':
    main()
