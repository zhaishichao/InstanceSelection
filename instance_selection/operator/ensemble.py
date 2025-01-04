# 集成分类器的投票
import numpy as np
from scipy.stats import mode
from sklearn.base import clone
from utils.dataset_utils import get_subset

# 集成分类器的预测结果
def vote_result_ensembles(ensembles, x_test, result='soft'):
    y_pred_labels_ensembles = []
    y_pred_prob_labels_ensembles = []
    for ensemble in ensembles:
        ind_pred = ensemble.predict(x_test)  # 计算accuracy、PPV
        ind_proba = ensemble.predict_proba(x_test)
        y_pred_labels_ensembles.append(ind_pred)
        y_pred_prob_labels_ensembles.append(ind_proba)
    # 按列投票，取每列中出现次数最多的类别作为最终分类结果
    vote_pred = mode(y_pred_labels_ensembles, axis=0, keepdims=False).mode.flatten()
    # 堆叠为三维数组
    vote_pred_prob = np.stack(y_pred_prob_labels_ensembles, axis=0)
    # 对第一个维度 (num_classifiers) 求平均
    vote_pred_prob = np.mean(vote_pred_prob, axis=0)
    if result == 'soft':
        return vote_pred_prob
    elif result == 'hard':
        return vote_pred
    else:
        raise ValueError('result must be "soft" or "hard"')  # 提示出错

# 集成个体
def ensemble_individuals(individuals, model, x_train, y_train):
    ensembles = []
    for ind in individuals:
        x_sub, y_sub = get_subset(ind, x_train, y_train)
        # 用实例选择的子集训练模型
        model_clone = clone(model)
        model_clone.fit(x_sub, y_sub)
        ind.model = model_clone
        ensembles.append(ind.model)
    return ensembles
