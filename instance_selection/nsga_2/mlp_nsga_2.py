from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils.dataset_utils import get_classes_indexes_counts
import scipy.io as sio  # 从.mat文件中读取数据集
import numpy as np

################################################################加载数据集################################################
# 数据集
mat_data = sio.loadmat('../../data/dataset/Connect4.mat')
# 提取变量
dataset_x = mat_data['X']
dataset_y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
# 显示数据集分布
print("特征数据:", dataset_x.shape)
print("label:", dataset_y.shape)
# 统计每个类别的个数，dataset_y.max()+1是类别的个数
classes, counts = get_classes_indexes_counts(dataset_y)
print("每种类别的数量：", counts)

########################################################################################################################
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=42)
# 显示数据集分布
print("特征数据:", x_train.shape)
print("label:", y_train.shape)
# 统计每个类别的个数
classes_train, counts_train = get_classes_indexes_counts(y_train)
print("每种类别的数量：", counts_train)

########################################################################################################################
# 确定每个类别的数量
num_instances = int(counts_train.min() * 0.9)  # 向下取整
print("最小数量:", num_instances)

# 在每个类别中随机的选择该数量的实例的索引
balanced_classes = np.array([])
for indexes in classes_train:
    random_selecte_indices = np.random.choice(indexes, size=num_instances, replace=False)
    balanced_classes = np.hstack((balanced_classes, random_selecte_indices))
balanced_classes = np.sort(balanced_classes).astype(int)

# 得到平衡的数据集
balanced_dataset_x = []
balanced_dataset_y = np.array([])
for index in balanced_classes:
    balanced_dataset_x.append(x_train[index])
    balanced_dataset_y = np.hstack((balanced_dataset_y, y_train[index]))
balanced_dataset_x = np.array(balanced_dataset_x)
balanced_dataset_y = np.array(balanced_dataset_y).astype(int)

# 显示数据集分布
print("平衡的数据集的特征数据:", balanced_dataset_x.shape)
print("label:", balanced_dataset_y.shape)

# 统计每个类别的个数
classes_balanced_dataset, counts_balanced_dataset = get_classes_indexes_counts(balanced_dataset_y)
print("平衡的数据集中每种类别的数量：", counts_balanced_dataset)






########################################################################################################################
from sklearn.metrics import confusion_matrix, precision_score


##########################由个体得到选择的实例子集的索引###########################
def get_indices(individual):
    '''
    :param individual: individual（用实值进行编码）
    :return: 被选择实例的索引
    '''
    individual = np.round(individual)  # 数据范围在0-1之间，转化成int的同时会舍去小数部分，从而将个体映射到0-1编码
    indices = np.where(individual == 1)  # 1代表选择该实例，返回值是tuple，tuple[0]取元组中的第一个元素
    return indices[0]


###########################获取实例子集############################
def get_subset(individual):
    '''
    :param individual:
    :return: 实例子集
    '''
    indices = get_indices(individual)
    x_sub = balanced_dataset_x[indices, :]
    y_sub = balanced_dataset_y[indices]
    return x_sub, y_sub


##########################适应度函数（PPV和PFC，为主要、次要指标）#################################
def fitness_function(x_sub, y_sub, ensembles, index):
    ######################PPV#######################
    # 使用训练数据进行预测
    index_pred = ensembles[index].predict(x_test)
    # 计算混淆矩阵 average="micro"也即PPV，每个类别的tp/(tp+fp)
    ppv = precision_score(y_test, index_pred, average="micro")

    ######################PFC#######################
    f2 = 0.0
    for i in range(len(ensembles)):
        if i != index:
            # 计算两个数组中索引对应的元素值不相等的个数
            i_pred = ensembles[i].predict(x_test)
            # 每个类别的错误数可以通过 np.sum(conf_matrix, axis=1) - np.diag(conf_matrix) 得到，这个操作计算了每一行的总和减去对角线的正确预测数，即为错误分类数。
            # 计算混淆矩阵
            conf_matrix = confusion_matrix(y_test, i_pred)
            # 计算每个类别的分类错误数
            classification_errors = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
            classification_errors_counts = np.sum(classification_errors)
            if classification_errors_counts == 0:
                classification_errors_counts = 1
            count = sum(1 for a, b in zip(index_pred, i_pred) if a != b)
            f2 = f2 + count / classification_errors_counts
    pfc = f2 / (len(ensembles) - 1)
    return round(ppv, 4), round(pfc, 4)

############################################################################################################
import array
import random
import numpy
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMinAndMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMinAndMax)
toolbox = base.Toolbox()

# Problem definition

BOUND_LOW, BOUND_UP = 0.0, 1.0

NDIM = num_instances


def uniform(low, up, size=None):
    return [random.uniform(low, up) for i in range(size)]



toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("evaluate", fitness_function)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("select", tools.selTournament, tournsize=3)

init_mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42)


def main(seed=None):
    random.seed(seed)

    NGEN = 5
    MU = 30
    CXPB = 0.9

    ####################################迭代过程的记录###########################
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max"

    ####################################种群的初始化###########################
    pop = toolbox.population(n=MU)

    ####################################计算初始种群的适应度###########################
    ensembles = []  # 当前每个个体对应的mlp模型
    save_ensembles = []  # 存储每个个体对应的mlp模型
    pop_x_sub = []  # 当前每个个体的实例选择的特征数据
    pop_y_sub = []  # 当前每个个体对应的实例选择的lable
    # 对于每个个体都训练得到一个mlp模型
    for i in range(len(pop)):
        mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, random_state=42)
        x_sub, y_sub = get_subset(pop[i])
        mlp.fit(x_sub, y_sub)
        ensembles.append(mlp)
        pop_x_sub.append(x_sub)
        pop_y_sub.append(y_sub)
    save_ensembles = ensembles  # 保存初始种群对应的mlp集合
    # 由mlp模型得到个体的适应度
    for i in range(len(pop)):
        pop[i].fitness.values = toolbox.evaluate(pop_x_sub[i], pop_y_sub[i], ensembles, i)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)
    ####################################种群的迭代###########################
    for gen in range(1, NGEN):
        # 选择
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 交叉
        for i in range(0, len(offspring) - 1, 2):
            if random.random() <= CXPB:
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])
            # 突变
            offspring[i] = toolbox.mutate(offspring[i])[0]
            offspring[i + 1] = toolbox.mutate(offspring[i + 1])[0]
            del offspring[i].fitness.values, offspring[i + 1].fitness.values

        # 计算新的种群适应度
        ensembles.clear()
        pop_x_sub.clear()
        pop_y_sub.clear()
        for i in range(len(offspring)):
            mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, random_state=42)
            x_sub, y_sub = get_subset(offspring[i])
            mlp.fit(x_sub, y_sub)
            ensembles.append(mlp)
            pop_x_sub.append(x_sub)
            pop_y_sub.append(y_sub)
        for i in range(len(offspring)):
            offspring[i].fitness.values = toolbox.evaluate(pop_x_sub[i], pop_y_sub[i], ensembles, i)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), **record)
        print(logbook.stream)

    return pop, logbook


if __name__ == "__main__":
    pop, stats = main()
