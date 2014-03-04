#coding=utf8
'''
计算指标所用的函数，如何使用请参考底部的main函数

'''


def predition2dict(pred):
    """ Transform predictions to dict
    """
    #import ipdb; ipdb.set_trace()
    _dict = {}
    n = len(pred)
    for i in range(n):
        key = str(i)
        if isinstance(pred[i], int):
            _dict[key] = [str(pred[i])] # multi-class
        else: 
            _dict[key] = [str(v) for v in pred[i, :]] # multi-label
            
    return _dict


# 说明：此函数用于解析上传文件的数据，封装为python的dict数据结构
# 参数：filename为文件名
def parse_file_v2(filename, closed=True):
    _dict = {}
    f = None
    if type(filename) is str:
        f = open(filename)
    else:
        f = filename

    line = f.readline().strip()
    while line:
        kv = line.split(',')
        _dict[kv[0]] = [v.strip() for v in kv[1:] if v.strip()]
        line = f.readline().strip()

    if hasattr(f, 'close') and closed:
        try:
            f.close()
        except:
            pass

    return _dict

# 说明：此函数用于计算average precision，使用了score_item_v2函数
# 参数：l为解析后的预测结果，al为解析后的真实结果
# 返回值：average precision
def score_list_v2(l, al):
    # 如果预测结果数目和真实结果数目不同，返回0
    if len(l)!=len(al):
        return 0.0
    ls = []
    # 遍历预测结果中的每一行
    for k,v in l.items():
        # 计算中间结果，返回每一行的中间结果ls[x][0]和该行是否有效的标记ls[x][1]
        ls.append( score_item_v2(v, al[k]) )
    # 加和中间结果
    sums = sum( i[0] for i in ls )

    # 计算有效instance数目
    lens = sum( i[1] for i in ls)

    # 结果
    if lens:
        return sums/lens
    return 0.0

# 说明：此函数用于计算average precision的中间结果
# 参数：item为预测结果中的某一行的类标部分，aitem为实际结果中的某一行的类标部分
# 返回：(x,y), x为改行分数，y表示该行是否有效，若无效则为0
def score_item_v2(item, aitem):
    # 变成浮点数
    item_f = [float(i) for i in item]

    # 建立索引号
    index = range(12)
    # 组成[(0,a),(1,b),(3,c),...]的形式，方便之后排序后仍知道原序列号
    item_sort = zip(index, item_f)
    # 对预测结果进行从小到大排序并获得排序后的索引号
    item_sort.sort(key=lambda x:x[1])
    #获得索引号的重新排列结果
    index_sort = [i[0] for i in item_sort]

    # 对实际结果进行相同排序
    aitem_f = [float(i) for i in aitem]
    aitem_sort = [aitem_f[i] for i in index_sort]

    #去除全0或全1的数据
    if sum(aitem_f)==12 or sum(aitem_f)==-12:
        return (0.0,0)

    # 计算每个类标的分数
    score = []
    item_size = len(aitem_sort)
    for i,v in enumerate(aitem_sort):
        if v == 1:
            # 前面等于1的个数+自己1个 / i+1自身位置+1;从1开始
            score.append(len([j for j in aitem_sort[i:] if j==1])*1.0/(item_size-i))

    # 计算行分数 该行每个的分数之和/该行的1的数目
    sums = sum(score)
    ones = len([i for i in aitem_sort if i==1])
    if ones:
        return (sums/ones,1)
    return (0.0,1)


# 说明：此函数用于计算三类分类问题的f1measure，独立计算每个类别的f1之后平均，例如计算0类的f1，这1类、2类为负类
# 参数：submit为解析后的预测结果，answer为解析后的实际结果
# 返回：f1measure
def score_list2(submit, answer):
    # 如果预测结果数目和真实结果数目不同，返回0
    if len(submit)!=len(answer):
        return 0.0

    #属于0类的instance的容器
    is0_p = []
    #属于1类的instance的容器
    is1_p = []
    #属于2类的instance的容器
    is2_p = []

    # 在实际结果中
    for a,v in answer.items():
        # 如果格式不正确返回0
        try:
            int(v[0])
        except:
            return 0.0
        # 属于0的装入一个数组
        if int(v[0]) == 0:
            is0_p.append(a)
        # 属于1的装入一个数组
        elif int(v[0]) == 1:
            is1_p.append(a)
        # 属于2的装入一个数组
        else:
            is2_p.append(a)

    # 设置tp和fp的初值
    is0_tp = 0
    is0_fp = 0
    is1_tp = 0
    is1_fp = 0
    is2_tp = 0
    is2_fp = 0

    # 在预测结果中
    for a,v in submit.items():
        # 如果格式不正确返回0
        try:
            int(v[0])
        except:
            return 0.0

        # 如果预测为0类
        if int(v[0]) == 0:
            # 如果真实类标为0
            if a in is0_p:
                #is0_tp加1
                is0_tp += 1
            else:
                #否则is0_fp加1
                is0_fp += 1

        # 1类同上
        elif int(v[0]) == 1:
            if a in is1_p:
                is1_tp += 1
            else:
                is1_fp += 1

        # 2类同上
        elif int(v[0]) == 2:
            if a in is2_p:
                is2_tp += 1
            else:
                is2_fp += 1

    # print is0_p, is0_tp, is0_fp, '==', is1_p, is1_tp, is1_fp, '==', is2_p, is2_tp, is2_fp, '=='

    #根据公式计算0类f1
    if is0_tp == 0:
        is0_precision = 0
        is0_recall = 0
        is0_f1 = 0
    else:
        is0_precision = is0_tp*1.0/(is0_tp+is0_fp)
        is0_recall = is0_tp*1.0/len(is0_p)
        is0_f1 = 2.0*is0_recall*is0_precision/(is0_recall+is0_precision)

    #根据公式计算1类f1
    if is1_tp == 0:
        is1_precision = 0
        is1_recall = 0
        is1_f1 = 0
    else:
        is1_precision = is1_tp*1.0/(is1_tp+is1_fp)
        is1_recall = is1_tp*1.0/len(is1_p)
        is1_f1 = 2.0*is1_recall*is1_precision/(is1_recall+is1_precision)

    #根据公式计算2类f1
    if is2_tp == 0:
        is2_precision = 0
        is2_recall = 0
        is2_f1 = 0
    else:
        is2_precision = is2_tp*1.0/(is2_tp+is2_fp)
        is2_recall = is2_tp*1.0/len(is2_p)
        is2_f1 = 2.0*is2_recall*is2_precision/(is2_recall+is2_precision)

    # print is0_precision, is0_recall, '===', is1_precision, is1_recall, '===', is2_precision, is2_recall
    #print '\nClass 0: precision = %f, recall = %f, F1 = %f' % (is0_precision, is0_recall, is0_f1)
    #print 'Class 1: precision = %f, recall = %f, F1 = %f' % (is1_precision, is1_recall, is1_f1)
    #print 'Class 2: precision = %f, recall = %f, F1 = %f\n' % (is2_precision, is2_recall, is2_f1)
    
    # 平均
    f1 = (is0_f1 + is1_f1 + is2_f1)/3
    return f1

if __name__ == '__main__':

    # load answer
    answer = parse_file_v2('example_task/example_task2.csv')

    # load predict
    submit = parse_file_v2('predict.csv')


    # cal average_precision
    average_precision = float(score_list_v2(submit, answer))*100

    # cal f1
    f1 = float(score_list2(submit, answer))*100

    print average_precision
    print f1

