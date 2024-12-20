import bisect
from collections import defaultdict
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random

import numpy as np
from sklearn.metrics import confusion_matrix


######################################
# 添加约束条件，得到种群中的不可行解和可行解 #
######################################
# 计算cv值
def CV(individual, Acc1, Acc2, Acc3):
    acc = (
        Acc1 - individual.fitness.values[0], Acc2 - individual.fitness.values[1], Acc3 - individual.fitness.values[2])
    cv = 0
    for i in range(len(acc)):
        # 求0和cv中的最大值之和
        cv = cv + max(0, acc[i])
        individual.fitness.cv = cv  # 将cv值保存在个体中
    return cv


# 获得可行解与不可行解
def get_feasible_infeasible(pop, Acc1, Acc2, Acc3):
    index = []
    for i in range(len(pop)):
        if CV(pop[i], Acc1, Acc2, Acc3) > 0:
            index.append(i)
    # 获取可行解与不可行解
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 得到可行解
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 得到不可行解
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"))  # 对种群中的不可行解按照个体的cv值升序排序
    return feasible_pop, infeasible_pop


######################################
#     适应度函数（Acc1,Acc2,Acc3）      #
######################################
def fitness_function(individual, weights_train):
    # 使用训练数据进行预测
    y_sub, ind_pred = individual.y_sub_and_pred[0], individual.y_sub_and_pred[1]  # 获取个体的实例选择标签和预测标签
    ######################计算混淆矩阵#########################
    cm = confusion_matrix(y_sub, ind_pred)
    tp_per_class = cm.diagonal()  # 对角线元素表示每个类预测正确的个数，对角线求和，即所有预测正确的实例个数之和，计算Acc1
    s_per_class = cm.sum(axis=1)
    Acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc1
    Acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc2
    Acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights_train)  # Acc3
    return round(Acc1, 4), round(Acc2, 4), round(Acc3, 4)


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
        select_class_indices = {element: np.random.choice(indices, num_instances, replace=False) for element, indices in
                                class_indices.items()}
        for element in unique_elements:
            for index in select_class_indices[element]:
                population[i][index] = 1
    return population


######################################
#      mutate(二进制随机反转)           #
######################################
def mutate_binary_inversion(individual, mutation_rate=0.2):
    num_genes = len(individual)  # 基因总数
    num_mutation = math.ceil(random.uniform(0.05, mutation_rate) * num_genes)  # 要突变的总数
    sampled_indices = random.sample(range(num_genes), num_mutation)  # 在num_genes个基因中随机采样num_mutation个
    for index in sampled_indices:
        if individual[index] == 0:
            individual[index] = 1
        else:
            individual[index] = 0
    return individual,


######################################
#    查重（找到种群中互相重复的个体）      #
######################################
# 得到重复个体（基于similar）
def find_duplicates_based_on_similarity(pop, similar=0.9):
    """
    找到重复个体的索引。
    :param arrays: 一个包含 array.array 的列表
    :param threshold: 重复的判断阈值
    :return: 重复对的索引列表
    """
    n = len(pop)
    duplicates = []  # 用于记录重复对的索引

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # 当前两组数组
            a = pop[i]
            b = pop[j]

            # 计算1的个数
            ones_a = sum(a)
            ones_b = sum(b)

            # 如果其中一个数组全是0，不可能满足条件
            if ones_a == 0 or ones_b == 0:
                continue

            # 计算交集中的1的数量
            common_ones = sum(x == 1 & y == 1 for x, y in zip(a, b))

            # 判断是否满足重复的定义
            if (common_ones / ones_a > similar) and (common_ones / ones_b > similar):
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)
    return duplicates


# 得到重复个体（完全一样）
def find_duplicates(pop):
    """
    找到重复个体的索引。
    :param arrays: 一个包含 array.array 的列表
    :param threshold: 重复的判断阈值
    :return: 重复对的索引列表
    """
    n = len(pop)
    duplicates = []  # 用于记录重复对的索引

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # 计算pop[i],pop[j]之间的汉明距离
            hamming_distance = sum(x != y for x, y in zip(pop[i], pop[j]))
            if hamming_distance == 0:
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)
    return duplicates


######################################
#               去重                  #
######################################
# 根据索引对，去除种群中重复的个体
def remove_duplicates(pop, duplicates):
    """
    移除重复的个体。
    :param arrays: 一个包含 array.array 的列表
    :param duplicates: 重复对的索引列表
    :return: 去重后的列表
    """
    # 找到所有需要移除的索引
    to_remove = set()  # 只保留后出现的索引
    for duplicate in duplicates:
        to_remove.update(duplicate)  # update是用来更新set集合的
    # 构造去重后的列表
    return [pop[i] for i in range(len(pop)) if i not in to_remove], len(to_remove)


######################################
#    锦标赛选择，基于非支配排序和拥挤距离   #
######################################
def selRandom(individuals, k):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]


def selTournamentNDCD(individuals, k, tournsize):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    # 先做非支配排序，再根据选择支配等级进行选择
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)  # 随机选择tournsize个个体
        aspirants = sortNondominated(aspirants, len(aspirants))  # 进行非支配排序
        aspirants_rank_first = sorted(aspirants[0], key=attrgetter("fitness.crowding_dist"),
                                      reverse=True)  # 在第一个等级内按cv升序排列
        chosen.append(aspirants_rank_first[0])  # 选择第一个等级中cv约束最小的
    return chosen


######################################
# Non-Dominated Sorting   (NSGA-II)  #
######################################

def selNSGA2(individuals, k, nd='standard', x_test=None, y_test=None):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi_objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    # for front in pareto_fronts:
    # assignCrowdingDist(front, x_test, y_test)
    # assignCrowdingDist(individuals, x_test, y_test)
    assignCrowdingDist_PFC(individuals, x_test, y_test)
    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen, pareto_fronts


def sortNondominated(individuals, k, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi_objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []
    return fronts


def assignCrowdingDist_PFC(individuals, x_test, y_test):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return
    pred_lists = []  # 每个个体的预测结果， 0表示对应实例预测错误，1表示预测正确
    for ind in individuals:
        y_pred = ind.mlp.predict(x_test)  # 模型预测结果
        binary_list = [1 if x == y else 0 for x, y in zip(y_pred, y_test)]  # 0表示对应实例预测错误，1表示预测正确
        pred_lists.append(binary_list)

    accfailcred = [[0 for _ in range(len(individuals))] for _ in range(len(individuals))]  #
    for i in range(len(individuals) - 1):
        count_zeros_i = sum(1 for x in pred_lists[i] if x == 0)
        for j in range(i + 1, len(individuals)):
            # 计算汉明距离
            hamming_distance = sum(x != y for x, y in zip(pred_lists[i], pred_lists[j]))
            count_zeros_j = sum(1 for x in pred_lists[j] if x == 0)
            accfailcred[i][j] = 1.0 * hamming_distance / (count_zeros_i + count_zeros_j)
            accfailcred[j][i] = accfailcred[i][j]
    # 对每一行求和
    row_sum_accfailcred = [sum(row) for row in accfailcred]
    for i in range(len(individuals)):
        individuals[i].fitness.crowding_dist = 1.0 * row_sum_accfailcred[i] / (len(individuals) - 1)  # 使用PFC代替拥挤距离


def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist


def selTournamentDCD(individuals, k):
    """Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4 only if k is equal to the length of individuals.
    Starting from the beginning of the selected individuals, two consecutive
    individuals will be different (assuming all individuals in the input list
    are unique). Each individual from the input list won't be selected more
    than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select. Must be less than or equal
              to len(individuals).
    :returns: A list of selected individuals.
    """

    if k > len(individuals):
        raise ValueError("selTournamentDCD: k must be less than or equal to individuals length")

    if k == len(individuals) and k % 4 != 0:
        raise ValueError("selTournamentDCD: k must be divisible by four if k == len(individuals)")

    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen


#######################################
# Generalized Reduced runtime ND sort #
#######################################


def identity(obj):
    """Returns directly the argument *obj*.
    """
    return obj


def isDominated(wvalues1, wvalues2):
    """Returns whether or not *wvalues2* dominates *wvalues1*.

    :param wvalues1: The weighted fitness values that would be dominated.
    :param wvalues2: The weighted fitness values of the dominant.
    :returns: :obj:`True` if wvalues2 dominates wvalues1, :obj:`False`
              otherwise.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
        if self_wvalue > other_wvalue:
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal


def median(seq, key=identity):
    """Returns the median of *seq* - the numeric value separating the higher
    half of a sample from the lower half. If there is an even number of
    elements in *seq*, it returns the mean of the two middle values.
    """
    sseq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sseq[(length - 1) // 2])
    else:
        return (key(sseq[(length - 1) // 2]) + key(sseq[length // 2])) / 2.0


def sortLogNondominated(individuals, k, first_front_only=False):
    """Sort *individuals* in pareto non-dominated fronts using the Generalized
    Reduced Run-Time Complexity Non-Dominated Sorting Algorithm presented by
    Fortin et al. (2013).

    :param individuals: A list of individuals to select from.
    :returns: A list of Pareto fronts (lists), with the first list being the
              true Pareto front.
    """
    if k == 0:
        return []

    # Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    for i, ind in enumerate(individuals):
        unique_fits[ind.fitness.wvalues].append(ind)

    # Launch the sorting algorithm
    obj = len(individuals[0].fitness.wvalues) - 1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)

    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA(fitnesses, obj, front)

    # Extract individuals from front list here
    nbfronts = max(front.values()) + 1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    # Keep only the fronts required to have k individuals.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:
                return pareto_fronts[:i + 1]
        return pareto_fronts
    else:
        return pareto_fronts[0]


def sortNDHelperA(fitnesses, obj, front):
    """Create a non-dominated sorting of S on the first M objectives"""
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        # Only two individuals, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if isDominated(s2[:obj + 1], s1[:obj + 1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        # All individuals for objective M are equal: go to objective M-1
        sortNDHelperA(fitnesses, obj - 1, front)
    else:
        # More than two individuals, split list and then apply recursion
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA(best, obj, front)
        sortNDHelperB(best, worst, obj - 1, front)
        sortNDHelperA(worst, obj, front)


def splitA(fitnesses, obj):
    """Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = [], []
    best_b, worst_b = [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b


def sweepA(fitnesses, front):
    """Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair] + 1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)


def sortNDHelperB(best, worst, obj, front):
    """Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called."""
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        # One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        # One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if isDominated(hi[:obj + 1], li[:obj + 1]) or hi[:obj + 1] == li[:obj + 1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        # All individuals from L dominate H for objective M:
        # Also supports the case where every individuals in L and H
        # has the same value for the current objective
        # Skip to objective M-1
        sortNDHelperB(best, worst, obj - 1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best, worst, obj)
        sortNDHelperB(best1, worst1, obj, front)
        sortNDHelperB(best1, worst2, obj - 1, front)
        sortNDHelperB(best2, worst2, obj, front)


def splitB(best, worst, obj):
    """Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)

    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b


def sweepB(best, worst, front):
    """Adjust the rank number of the worst fitnesses according to
    the best fitnesses on the first two objectives using a sweep
    procedure.
    """
    stairs, fstairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, False)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair] + 1)


__all__ = ['selNSGA2', 'sortNondominated', 'sortLogNondominated',
           'selTournamentDCD']
