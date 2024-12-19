import numpy as np


# 得到

# 得到分类、以及分类所对应的索引
# 也可使用numpy
# 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
# unique_elements, counts, indices = np.unique(labels, return_counts=True, return_inverse=True)
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


# 得到分类、以及分类所对应的数量，初始化个体为平衡数据集
def init_population_for_balanced_dataset(population, y, ratio):
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, counts = np.unique(y, return_counts=True)

    num_instances = int(np.ceil(counts.min() * ratio))
    # 构造每个类别的索引列表
    class_indices = {element: np.where(y == element)[0] for element in unique_elements}

    for i in range(len(population)):
        # 对于每个类，随机选择 num_instances 个不同的索引，生成一个新的dict
        select_class_indices = {element: np.random.choice(indices, num_instances, replace=False) for element, indices in
                                class_indices.items()}
        for element in unique_elements:
            for index in select_class_indices[element]:
                population[i][index]=1
    return population

def get_counts(y, output=False):
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


# 引用自
# [E. R. Q. Fernandes, A. C. P. L. F. de Carvalho and X. Yao
# “Ensemble of Classifiers Based on Multiobjective Genetic Sampling for Imbalanced Data,”
# in IEEE Transactions on Knowledge and Data Engineering, vol. 32, no. 6, pp. 1104-1115, 1 June 2020
# doi: 10.1109/TKDE.2019.2898861.]

# 1、得到平衡数据集
########################################################################################################################
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils.dataset_utils import get_classes_indexes_counts, balanced_dataset
import scipy.io as sio  # 从.mat文件中读取数据集
from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def datasetload(datasetname):
    # 随机种子
    random_seed = 42
    print("#########################加载数据集#########################")
    # 数据集
    # Nursery(0.1)、Satellite(0.001)、Contraceptive(0.1)
    # datasetname = 'Chess4.mat'
    mat_data = sio.loadmat('../../data/dataset/' + datasetname)

    dataset_x = mat_data['X']
    dataset_y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
    # 显示数据集分布
    print("特征数据:", dataset_x.shape)
    print("label:", dataset_y.shape)
    # 统计每个类别的个数
    classes, counts = get_classes_indexes_counts(
        dataset_y)  # np.argmax(y_onehot, axis=1)找最大值的索引，将0-1序列转化为0,1,2,3......的整数标签
    print("每种类别的分布：", counts)
    print("#########################划分数据集#########################")
    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=random_seed)

    # 数据的标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 显示数据集分布
    print("特征数据:", x_train.shape)
    print("label:", y_train.shape)
    # 统计每个类别的个数
    classes_train, counts_train = get_classes_indexes_counts(y_train)
    # 计算每个类的权重
    weights_train = (1 / counts_train.astype(float)) / np.sum(1 / counts_train.astype(float))
    print("训练集每种类别的分布：", counts_train)
    print("训练集每种类别的权重：", weights_train)
    classes_test, counts_test = get_classes_indexes_counts(y_test)
    print("测试集每种类别的分布：", counts_test)
    print("#########################平衡数据集#########################")
    # 确定每个类别的分布
    num_instances = int(counts_train.min() * 1.0)  # 向下取整
    num_instances_train = len(y_train)  # 取训练集的数量
    print("最小数量:", num_instances)
    # 在每个类别中随机的选择该数量的实例的索引
    balanced_dataset_x, balanced_dataset_y = balanced_dataset(x_train, y_train, num_instances)
    balanced_dataset_x = np.array(balanced_dataset_x)
    balanced_dataset_y = np.array(balanced_dataset_y).astype(int)
    # 显示数据集分布
    print("平衡的数据集的特征数据:", balanced_dataset_x.shape)
    print("label:", balanced_dataset_y.shape)
    # 统计每个类别的分布
    classes_balanced_dataset, counts_balanced_dataset = get_classes_indexes_counts(balanced_dataset_y)
    print("平衡的数据集中每种类别的分布：", counts_balanced_dataset)
