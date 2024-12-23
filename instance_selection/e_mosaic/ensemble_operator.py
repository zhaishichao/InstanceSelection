# 集成分类器的投票
import numpy as np
from scipy.stats import gmean, mode
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report


def vote_ensembles(ensembles, x_test, y_test, show_result=False):
    y_pred_labels_ensembles = []
    y_pred_prob_labels_ensembles = []
    for ensemble in ensembles:
        ind_pred = ensemble.predict(x_test)  # 计算accuracy、PPV
        ind_proba = ensemble.predict_proba(x_test)
        y_pred_labels_ensembles.append(ind_pred)
        y_pred_prob_labels_ensembles.append(ind_proba)
    # 按列投票，取每列中出现次数最多的类别作为最终分类结果
    final_pred_result = mode(y_pred_labels_ensembles, axis=0, keepdims=False).mode.flatten()
    # 堆叠为三维数组
    stacked_predictions_prob = np.stack(y_pred_prob_labels_ensembles, axis=0)
    # 对第一个维度 (num_classifiers) 求平均
    ensemble_predictions_prob = np.mean(stacked_predictions_prob, axis=0)
    # 计算 ROC AUC（ovo+macro）、G-Mean、recall_per_class
    geometric_mean, auc_ovo_macro, recall_per_class = calculate_gmean_mauc(ensemble_predictions_prob, y_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, final_pred_result)
    if show_result:
        print(f'Accuracy: {accuracy:.2f}')
        # 打印分类报告
        print("Classification Report:")
        print(classification_report(y_test, final_pred_result))
        # 打印混淆矩阵
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, final_pred_result))
    return round(geometric_mean, 4), round(auc_ovo_macro, 4), np.array(
        ["{:.4f}".format(x) for x in recall_per_class]).tolist()  # 保留六位小数


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


def ensembles_individuals_gmean_mauc(individuals):
    num_ensembles = len(individuals)
    sum_gmean = 0
    sum_mauc = 0
    for ind in individuals:
        sum_mauc = sum_mauc + ind.mauc
        sum_gmean = sum_gmean + ind.gmean
    # 求平均值
    avg_gmean = sum_gmean / num_ensembles
    avg_mauc = sum_mauc / num_ensembles
    return round(avg_gmean, 4), round(avg_mauc, 4)
