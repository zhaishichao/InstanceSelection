import numpy as np
from scipy.stats import gmean
from sklearn.metrics import confusion_matrix, roc_auc_score


######################################
#     适应度函数（Acc1,Acc2,Acc3）      #
######################################
def fitness_function(individual):
    # 使用训练数据进行预测
    y_sub, ind_pred_proba = individual.y_sub_and_pred_proba[0], individual.y_sub_and_pred_proba[1]  # 获取个体的实例选择标签和预测标签
    ind_pred = np.argmax(ind_pred_proba, axis=1)
    # 计算 ROC AUC（ovo+macro）、G-Mean、recall_per_class
    gmean, mauc, _ = calculate_gmean_mauc(ind_pred_proba, y_sub)
    return gmean, mauc
# 计算gmean,mauc
def calculate_gmean_mauc(y_pred_proba, y):
    # 计算 ROC AUC（ovo+macro）
    auc_ovo_macro = roc_auc_score(y, y_pred_proba, multi_class="ovo", average="macro")
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y, y_pred)
    # 计算每类召回率（每类正确预测个数 / 该类总数）
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    # 计算G-Mean
    geometric_mean = gmean(recall_per_class)
    return round(geometric_mean, 4), round(auc_ovo_macro, 4), recall_per_class

def calculate_accuracy(y_pred, y, weights_train):
    ######################计算混淆矩阵#########################
    cm = confusion_matrix(y, y_pred)
    tp_per_class = cm.diagonal()  # 对角线元素表示每个类预测正确的个数，对角线求和，即所有预测正确的实例个数之和，计算Acc1
    s_per_class = cm.sum(axis=1)
    Acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc1
    Acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc2
    Acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights_train)  # Acc3
    return round(Acc1, 4), round(Acc2, 4), round(Acc3, 4)