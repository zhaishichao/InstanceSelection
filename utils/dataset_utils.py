
import numpy as np

# 得到分类、以及分类所对应的索引
def get_classes_indexes_counts(y,output=False):
    # 统计每个类别的个数，y.max()+1是类别的个数
    num_class = y.max() + 1
    counts = np.zeros(num_class, dtype=int)
    classes = []
    for i in range(y.shape[0]):  # y.shape[0]相当于y的长度
        counts[y[i]] += 1
    for i in range(num_class):
        # np.where() 返回值是一个tuple数组，np.where(y == i)[0],表示取出该tuple数组的第一个元素，是一个ndarray数组
        classes.append(np.where(y == i)[0])
    if output :
        print("每种类别的数量：", counts)
    return classes, counts


# 得到分类、以及分类所对应的数量
def get__counts(y,output=False):
    # 统计每个类别的个数，y.max()+1是类别的个数
    num_class = y.max() + 1
    counts = np.zeros(num_class, dtype=int)
    for i in range(y.shape[0]):  # y.shape[0]相当于y的长度
        counts[y[i]] += 1
    if output :
        print("每种类别的数量：", counts)
    return counts