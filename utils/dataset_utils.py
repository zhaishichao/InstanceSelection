import numpy as np


# 得到分类、以及分类所对应的索引
def get_classes_indexes_counts(y, output=False):
    # 统计每个类别的个数，y.max()+1是类别的个数
    num_class = y.max() + 1
    counts = np.zeros(num_class, dtype=int)
    classes = []
    for i in range(y.shape[0]):  # y.shape[0]相当于y的长度
        counts[y[i]] += 1
    for i in range(num_class):
        # np.where() 返回值是一个tuple数组，np.where(y == i)[0],表示取出该tuple数组的第一个元素，是一个ndarray数组
        classes.append(np.where(y == i)[0])
    if output:
        print("每种类别的数量：", counts)
    return classes, counts


# 得到分类、以及分类所对应的数量
def get__counts(y, output=False):
    # 统计每个类别的个数，y.max()+1是类别的个数
    num_class = y.max() + 1
    counts = np.zeros(num_class, dtype=int)
    for i in range(y.shape[0]):  # y.shape[0]相当于y的长度
        counts[y[i]] += 1
    if output:
        print("每种类别的数量：", counts)
    return counts


##########################由个体得到选择的实例子集的索引###########################
def get_indices(individual):
    '''
    :param individual: individual（用二进制或0-1范围内的实值进行编码）
    :return: 被选择实例的索引
    '''
    individual = np.round(individual)  # 数据范围在0-1之间，转化成int的同时会舍去小数部分，从而将个体映射到0-1编码
    indices = np.where(individual == 1)  # 1代表选择该实例，返回值是tuple，tuple[0]取元组中的第一个元素
    return indices[0]


###########################获取实例子集############################
def get_subset(individual, dataset_x, dataset_y):
    '''
    :param individual:
    :return: 实例子集
    '''
    indices = get_indices(individual)
    x_sub = dataset_x[indices, :]
    y_sub = dataset_y[indices]
    return x_sub, y_sub


################################平衡数据集###########################
# 在每个类别中随机的选择该数量的实例的索引
def balanced_dataset(x_train, y_train, num_instances):
    balanced_classes = np.array([])
    classes_train, _ = get_classes_indexes_counts(y_train)
    for indexes in classes_train:
        random_selecte_indices = np.random.choice(indexes, size=num_instances, replace=False)
        balanced_classes = np.hstack((balanced_classes, random_selecte_indices))
    balanced_classes = np.sort(balanced_classes).astype(int)
    # 得到平衡的数据集
    balanced_dataset_x = []
    balanced_dataset_y = []
    for index in balanced_classes:
        balanced_dataset_x.append(x_train[index])
        balanced_dataset_y.append(y_train[index])
    balanced_dataset_x = np.array(balanced_dataset_x)
    balanced_dataset_y = np.array(balanced_dataset_y).astype(int)
    return balanced_dataset_x, balanced_dataset_y
