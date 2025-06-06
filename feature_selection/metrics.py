import numpy as np
from scipy.stats import gmean
from sklearn.metrics import confusion_matrix, roc_auc_score,f1_score



def calculate_gmean_mauc_f1(y_pred_proba, y):
    # 计算 ROC AUC（ovo+macro）
    # 统计y中有多少类
    num_classes = len(np.unique(y))
    if num_classes == 2:
        auc_ovo_macro = roc_auc_score(y, y_pred_proba[:, 1])
    else:
        auc_ovo_macro = roc_auc_score(y, y_pred_proba, multi_class="ovo", average="macro")
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y, y_pred)
    # 计算每类召回率（每类正确预测个数 / 该类总数）
    # 遍历混淆矩阵，计算每个类的召回率
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    # 计算G-Mean
    geometric_mean = gmean(recall_per_class)
    # 使用sklearn.metrics.f1_score计算F1-score
    f1 = f1_score(y, y_pred, average="macro")
    return round(geometric_mean, 6), round(auc_ovo_macro, 6),  round(f1, 6)

