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
# 得到分类、以及分类所对应的数量，初始化个体为平衡数据集
def init_population_for_balanced_dataset_2(population, y_train, ratio, show_details=False):
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, counts = np.unique(y_train, return_counts=True)
    num_instances = int(np.ceil(counts.min() * ratio))
    # 构造每个类别的索引列表
    class_indices = {element: np.where(y_train == element)[0] for element in unique_elements}
    if show_details:
        # 输出类别和类别对应的数量
        # 遍历class_indices,输出每个类，以及每个类的数量，以及索引
        for element in unique_elements:
            print(f"类别: {element}, 个数: {len(class_indices[element])}")
    for i in range(len(population)):
        # 对于每个类，随机选择 num_instances 个不同的索引，生成一个新的dict
        # 在num_instances和counts中对应的实例数量之间随机生成一个数字

        # select_class_indices = {element: np.random.choice(indices, num_instances, replace=False) for element, indices in
        #                         class_indices.items()}
        select_class_indices = {}
        for index, item in enumerate(class_indices.items()):
            random_number = random.randint(num_instances, counts[index])
            selected_indices = np.random.choice(item[1], random_number, replace=False)
            select_class_indices[item[0]] = selected_indices
        for element in unique_elements:
            for indexs in select_class_indices[element]:
                population[i][indexs] = 1
    return population


def init_population_for_balanced_dataset(population, y_train, ratio, show_details=False):
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, counts = np.unique(y_train, return_counts=True)
    num_instances = int(np.ceil(counts.min() * ratio))
    # 构造每个类别的索引列表
    class_indices = {element: np.where(y_train == element)[0] for element in unique_elements}
    if show_details:
        # 输出类别和类别对应的数量
        # 遍历class_indices,输出每个类，以及每个类的数量，以及索引
        for element in unique_elements:
            print(f"类别: {element}, 个数: {len(class_indices[element])}")
    for i in range(len(population)):
        # 对于每个类，随机选择 num_instances 个不同的索引，生成一个新的dict
        # 在num_instances和counts中对应的实例数量之间随机生成一个数字
        select_class_indices = {element: np.random.choice(indices, num_instances, replace=False) for element, indices in
                                class_indices.items()}
        for element in unique_elements:
            for indexs in select_class_indices[element]:
                population[i][indexs] = 1
    return population