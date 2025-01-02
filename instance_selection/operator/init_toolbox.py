from array import array

from deap import creator, base, tools

from instance_selection.operator.duplicate_process import find_duplicates, remove_duplicates
from instance_selection.operator.fitness import calculate_fitness
from instance_selection.operator.genetic_operator import mutate_binary_inversion
from instance_selection.operator.init_population import init_by_one_or_zero, init_population_based_balanced_method


def init_toolbox(y_train):
    NDIM = len(y_train)  # 个体基因长度
    # 最大化评价目标
    creator.create("FitnessMaxAndMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMaxAndMax, pfc=None, model=None,
                   y_sub_and_pred_proba=None, gmean=None, mauc=None)
    toolbox = base.Toolbox()
    toolbox.register("attr_binary", init_by_one_or_zero, binary=0)  # 0-1编码，基因全部初始化编码为0或1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_binary, n=NDIM)  # 个体初始化
    toolbox.register("init_population", init_population_based_balanced_method, y_train=y_train,
                     ratio=0.9)  # 初始化为平衡数据集（实例个数为min*0.9）
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 种群初始化
    toolbox.register("evaluate", calculate_fitness)  # 评价函数
    toolbox.register("mate", tools.cxOnePoint)  # 交叉
    toolbox.register("mutate", mutate_binary_inversion)  # 二进制突变
    toolbox.register("select", tools.selNSGA2)  # NSGA-II选择（非支配排序后）
    toolbox.register("find_duplicates", find_duplicates)  # 找到种群中重复个体的索引对
    toolbox.register("remove_duplicates", remove_duplicates)  # 去重
    return toolbox
