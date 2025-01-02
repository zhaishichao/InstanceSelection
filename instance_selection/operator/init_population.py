import random

import numpy as np


######################################
#         二项分布生成0-1序列           #
######################################

def exponential_distribution(lambda_, threshold):
    '''
    :param lambda_: 指数分布的参数λ（lambda）
    :param threshold: 阈值（阈值决定了生成0或1）
    :return:
    '''
    # 生成一个指数分布的随机数
    value = random.expovariate(lambda_)
    # 根据值与阈值的比较，生成 0 或 1
    if value < threshold:
        return 1
    else:
        return 0


# 全部初始化为0
def init_by_one_or_zero(binary=0):
    '''
    :param binary: 0或1
    :return: binary
    '''
    return binary

# 平衡数据集(只在初始化个体时使用)
def init_population_based_balanced_method(population, y_train, ratio, balanced_method='balanced'):
    '''

    :param population: 要进行初始化的种群
    :param y_train: 原始的训练集
    :param ratio: 平衡的比例（即初始实例数量的占比）
    :param balanced_method: 'balanced' or 'random'
    :return: population
    '''
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, counts = np.unique(y_train, return_counts=True)
    num_instances = int(np.ceil(counts.min() * ratio))
    # 构造每个类别的索引列表
    class_indices = {element: np.where(y_train == element)[0] for element in unique_elements}
    for i in range(len(population)):
        # 对于每个类，随机选择 num_instances 个不同的索引，生成一个新的dict
        select_class_indices = {}
        if balanced_method == 'balanced':
            select_class_indices = {element: np.random.choice(indices, num_instances, replace=False) for
                                    element, indices in class_indices.items()}
        elif balanced_method == 'random': #
            for index, item in enumerate(class_indices.items()):
                # 在num_instances和counts中对应的实例数量之间随机生成一个数字
                random_number = random.randint(num_instances, counts[index])
                selected_indices = np.random.choice(item[1], random_number, replace=False)
                select_class_indices[item[0]] = selected_indices
        else:
            raise ValueError("Invalid balanced_method")# 抛出异常
        for element in unique_elements:
            for indexs in select_class_indices[element]:
                population[i][indexs] = 1
    return population