
import numpy as np
from sklearn.metrics import confusion_matrix

######################################
#     适应度函数（Acc1,Acc2,Acc3）      #
######################################
def fitness_function(individual, weights_train):
    # 使用训练数据进行预测
    y_sub, ind_pred_proba = individual.y_sub_and_pred_proba[0], individual.y_sub_and_pred_proba[1]  # 获取个体的实例选择标签和预测标签
    ind_pred = np.argmax(ind_pred_proba, axis=1)
    ######################计算混淆矩阵#########################
    cm = confusion_matrix(y_sub, ind_pred)
    tp_per_class = cm.diagonal()  # 对角线元素表示每个类预测正确的个数，对角线求和，即所有预测正确的实例个数之和，计算Acc1
    s_per_class = cm.sum(axis=1)
    Acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc1
    Acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc2
    Acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights_train)  # Acc3
    return round(Acc1, 4), round(Acc2, 4), round(Acc3, 4)
