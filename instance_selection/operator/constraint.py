from operator import attrgetter

import numpy as np

from utils.dataset_utils import get_subset


######################################
# 添加约束条件，得到种群中的不可行解和可行解 #
######################################

# 计算cv值
def cv(individual, constraints):
    ind_fitness = individual.fitness.values
    if len(ind_fitness) != len(constraints):
        raise ValueError("约束条件和个体适应度无法匹配！")
    difference = [max(0, x - y) for x, y in zip(constraints, ind_fitness)]  # 判断个体适应度是否都大于等于约束条件
    cv = sum(difference)  # 求0和cv中的最大值之和，cv=0，表示是一个可行解
    individual.fitness.cv = cv  # 将cv值保存在个体中，用于排序
    return cv


# 获得可行解与不可行解
def get_feasible_infeasible(pop, constraints):
    index = []
    for i in range(len(pop)):
        if cv(pop[i], constraints) > 0:  # 判断个体适应度是否都大于等于约束条件
            index.append(i)  # 将不符合约束条件的个体的索引添加到index中
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 得到可行解
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 得到不可行解
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"))  # 对种群中的不可行解按照个体的cv值升序排序
    return feasible_pop, infeasible_pop


######################################
# 多分类：限制每个类至少有一个实例被选择  #
######################################

# 限制每个类至少有一个实例被选择
def individuals_constraints_in_classes(individuals, x_train, y_train, min_samples=5):
    '''
    如果存在未选择的类别；
    则在1-length（该类实例个数）之间生成一个随机数random_number；
    选择random_number个实例，添加到当前个体中。
    :param individuals: 每个个体
    :param x_train: 特征数据，在这个地方没什么用处，只是为了调用get_subset而已
    :param y_train: 标签
    :return: individuals （符合约束条件的个体集）
    '''
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, _ = np.unique(y_train, return_counts=True)
    # 构造每个类别的索引列表
    class_indices = {element: np.where(y_train == element)[0] for element in unique_elements}
    # 将unique_elements中的元素，构造一个set集合
    unique_elements_set = set(unique_elements)
    for individual in individuals:
        # 获取实例子集
        _, y_sub = get_subset(individual, x_train, y_train)
        unique_elements_sub, counts_sub = np.unique(y_sub, return_counts=True)
        unique_elements_sub_set = set(unique_elements_sub)
        # 对两个集合做差，得到差集（即未选择的类标签）
        unselected_set = unique_elements_set - unique_elements_sub_set
        # 如果差集不为空，则表示存在类没有被选择
        for selected_element, count in zip(unique_elements_sub, counts_sub):
            if count < min_samples:
                # 将该类标签添加到unselected_set中
                unselected_set.add(selected_element)
        if len(unselected_set) > 0:
            for unselected in unselected_set:
                # 获取unselected类的训练集的实例个数
                length = int(np.ceil(len(class_indices[unselected]) * 0.9))
                # 在0-length之间随机生成一个数字
                random_number = np.random.randint(5, length)
                selected_indices = np.random.choice(class_indices[unselected], random_number, replace=False)
                for index in selected_indices:
                    individual[index] = 1
    return individuals


def ensure_min_samples(individual, y_train, min_samples=3):
    """
    确保ind中选择的实例，每个类别至少有min_samples个，
    若不足，则从未选择的实例中补充。

    :param X: 实例数据，形状为 (n_samples, n_features)
    :param Y: 标签数据，形状为 (n_samples,)
    :param ind: 选择序列，形状为 (n_samples,)
    :param min_samples: 每个类别的最小选择样本数
    :return: 更新后的选择序列 ind
    """
    Y = np.array(y_train)

    unique_classes = np.unique(Y)

    for cls in unique_classes:
        selected_indices = np.where((individual == 1) & (Y == cls))[0]
        unselected_indices = np.where((individual == 0) & (Y == cls))[0]

        if len(selected_indices) < min_samples:
            num_needed = min_samples - len(selected_indices)
            if len(unselected_indices) >= num_needed:
                additional_indices = np.random.choice(unselected_indices, num_needed, replace=False)
            else:
                additional_indices = unselected_indices  # 如果不足，则全选
            # 将additional_indices转换成整数
            additional_indices = additional_indices.astype(int)
            for ind in additional_indices:
                individual[ind] = 1

    return individual
