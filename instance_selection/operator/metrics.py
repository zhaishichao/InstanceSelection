import numpy as np
from scipy.stats import gmean
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.base import clone

from utils.dataset_utils import get_subset, k_fold_cross_validation


# 计算适应度，同时会保存训练好的mlp模型
def evaluate_individuals(individuals, model, x_train, y_train, n_splits, random_seed, fitness_function=None):
    '''

    :param individuals:individuals
    :param model: 集成模型
    :param x_train: 训练集特征数据
    :param y_train: 训练集标签
    :param n_splits: k-fold的参数k
    :param random_seed: 随机种子
    :param fitness_function: 该参数是一个函数名，用来计算个体的适应度
    :return:
    '''
    for individual in individuals:
        x_sub, y_sub = get_subset(individual, x_train, y_train)
        y_pred_proba = k_fold_cross_validation(model=clone(model), X=x_sub, y=y_sub, n_splits=n_splits, method='soft',
                                               random_state=random_seed)  # 交叉验证得到软标签
        individual.y_sub_and_pred_proba = (y_sub, y_pred_proba)  # 保存个体的软标签和预测概率
        individual.gmean, individual.mauc, _ = calculate_gmean_mauc(y_pred_proba, y_sub)  # 计算个体的gmean和mauc
        if not individual.fitness.valid:
            individual.fitness.values = fitness_function(individual)  # 计算个体的目标值

######################################
#              适应度函数              #
######################################
def fitness_gmean_mauc(individual):
    y_sub, ind_pred_proba = individual.y_sub_and_pred_proba[0], individual.y_sub_and_pred_proba[1]  # 获取个体的实例选择标签和预测概率
    gmean, mauc, _ = calculate_gmean_mauc(ind_pred_proba, y_sub)  # 计算 ROC AUC（ovo+macro）、G-Mean、recall_per_class
    return gmean, mauc


def fitness_accuracy(individual, weights_train):
    y_sub, ind_pred_proba = individual.y_sub_and_pred_proba[0], individual.y_sub_and_pred_proba[1]  # 获取个体的实例选择标签和预测概率
    ind_pred = np.argmax(ind_pred_proba, axis=1)  # 获取个体的预测标签
    Acc1, Acc2, Acc3 = calculate_accuracy(ind_pred, y_sub, weights_train)  # 计算 Acc1、Acc2、Acc3
    return Acc1, Acc2, Acc3


######################################
#            计算gmean、mauc          #
######################################
def calculate_gmean_mauc(y_pred_proba, y):
    # 计算 ROC AUC（ovo+macro）
    auc_ovo_macro = roc_auc_score(y, y_pred_proba, multi_class="ovo", average="macro")
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y, y_pred)
    # 计算每类召回率（每类正确预测个数 / 该类总数）
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    # 计算G-Mean
    geometric_mean = gmean(recall_per_class)
    return round(geometric_mean, 6), round(auc_ovo_macro, 6), recall_per_class


def calculate_average_gmean_mauc(individuals):
    len_ind = len(individuals)
    sum_gmean = 0
    sum_mauc = 0
    for ind in individuals:
        sum_gmean = sum_gmean + ind.gmean
        sum_mauc = sum_mauc + ind.mauc
    # 求平均值
    avg_gmean = sum_gmean / len_ind
    avg_mauc = sum_mauc / len_ind
    return round(avg_gmean, 6), round(avg_mauc, 6)


######################################
#         计算Acc1、Acc2、Acc3         #
######################################

def calculate_accuracy(y_pred, y, weights):
    cm = confusion_matrix(y, y_pred)
    tp_per_class = cm.diagonal()  # 对角线元素表示每个类预测正确的个数，对角线求和，即所有预测正确的实例个数之和，计算Acc1
    s_per_class = cm.sum(axis=1)
    Acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc1
    Acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc2
    Acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights)  # Acc3
    return round(Acc1, 6), round(Acc2, 6), round(Acc3, 6)


def calculate_average_accuracy(individuals):
    len_ind = len(individuals)
    sum_Acc1 = 0
    sum_Acc2 = 0
    sum_Acc3 = 0
    for ind in individuals:
        sum_Acc1 = sum_Acc1 + ind.fitness.values[0]
        sum_Acc2 = sum_Acc2 + ind.fitness.values[1]
        sum_Acc3 = sum_Acc3 + ind.fitness.values[2]
    # 求平均值
    avg_Acc1 = sum_Acc1 / len_ind
    avg_Acc2 = sum_Acc2 / len_ind
    avg_Acc3 = sum_Acc3 / len_ind

    return round(avg_Acc1, 6), round(avg_Acc2, 6), round(avg_Acc3, 6)
