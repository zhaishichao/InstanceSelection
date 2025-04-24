import array

from deap import creator, base, tools

from instance_selection_encapsulation.operator.constraint import get_feasible_infeasible, individuals_constraints_in_classes, \
    ensure_min_samples
from instance_selection_encapsulation.operator.duplicate_process import remove_duplicates
from instance_selection_encapsulation.operator.metrics import fitness_accuracy, evaluate_individuals
from instance_selection_encapsulation.operator.genetic_operator import mutate_binary_inversion, selNSGA2
from instance_selection_encapsulation.operator.init_population import init_by_one_or_zero, init_population_based_balanced_method


def init_toolbox(model, x_train, y_train, weights_train, constraints, n_splits=5, random_seed=42):
    NDIM = len(y_train)  # 个体基因长度
    creator.create("FitnessMaxAndMax", base.Fitness, weights=(1.0, 1.0, 1.0))  # 最大化评价目标
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMaxAndMax, pfc=None, model=None, y_sub_and_pred_proba=None, gmean=None, mauc=None)
    toolbox = base.Toolbox()
    toolbox.register("attr_binary", init_by_one_or_zero, binary=0)  # 0-1编码，基因全部初始化编码为0或1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_binary, n=NDIM)  # 个体初始化
    toolbox.register("init_population", init_population_based_balanced_method, y_train=y_train, ratio=0.9)  # 初始化为平衡数据集（实例个数为min*0.9）
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 种群初始化
    toolbox.register("fitness_function", fitness_accuracy, weights_train=weights_train)
    toolbox.register("evaluate", evaluate_individuals, model=model, x_train=x_train, y_train=y_train, n_splits=n_splits,
                     random_seed=random_seed, fitness_function=toolbox.fitness_function)  # 评价个体
    toolbox.register("mate", tools.cxOnePoint)  # 交叉
    toolbox.register("mutate", mutate_binary_inversion)  # 二进制突变
    toolbox.register("select", selNSGA2)  # NSGA-II选择（非支配排序后）
    toolbox.register("individuals_constraints", individuals_constraints_in_classes, x_train=x_train, y_train=y_train)  # 限制每个类至少有一个实例被选择
    toolbox.register("individual_constraint", ensure_min_samples, y_train=y_train, min_samples=3)  # 限制每个类至少有一个实例被选择（对个体）
    toolbox.register("remove_duplicates", remove_duplicates, penalty_factor=0.0)  # 去重
    toolbox.register("get_feasible_infeasible", get_feasible_infeasible, constraints=constraints)  # 获取种群的可行解与不可行解
    return toolbox