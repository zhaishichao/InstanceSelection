from itertools import chain
import math
from operator import attrgetter
import random

from deap import tools

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
#    锦标赛选择，基于非支配排序和拥挤距离   #
######################################
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
        aspirants = tools.selRandom(individuals, tournsize)  # 随机选择tournsize个个体
        pareto_fronts = tools.sortNondominated(aspirants, len(aspirants))  # 进行非支配排序
        tools.assignCrowdingDist(pareto_fronts[0])
        pareto_first_front = sorted(pareto_fronts[0], key=attrgetter("fitness.crowding_dist"),
                                      reverse=True)  # 按拥挤度降序排列
        chosen.append(pareto_first_front[0])  # 选择第一个等级中拥挤度最大的
    return chosen


######################################
# Non-Dominated Sorting   (NSGA-II)  #
######################################

def selNSGA2(individuals, k, nd='standard', x_test=None, y_test=None):
    """
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi_objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = tools.sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = tools.sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    assignCrowdingDist_PFC(individuals, x_test, y_test)
    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen, pareto_fronts


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
