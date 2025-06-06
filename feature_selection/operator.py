import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_selection.metrics import calculate_gmean_mauc_f1


def non_dominated_sort(arr1, arr2, arr3):
    """
    对三个特征排名数组进行非支配排序，返回所有前沿
    参数:
        arr1, arr2, arr3: 一维ndarray数组，表示三个不同的特征排名
    返回:
        list of lists: 每个子列表代表一个前沿面，按Front 1, Front 2, ...排列
    """
    # 1. 合并三个数组的排名
    num_features = len(arr1)
    features = np.column_stack((arr1, arr2, arr3))
    # 2. 非支配排序
    domination_counts = np.zeros(num_features, dtype=int)
    dominated_features = [[] for _ in range(num_features)]
    fronts = [[]]  # fronts[0] = Front 1, fronts[1] = Front 2, ...
    # 计算支配关系
    for i in range(num_features):
        for j in range(i + 1, num_features):
            # 检查i是否支配j
            if np.all(features[i] <= features[j]) and np.any(features[i] < features[j]):
                dominated_features[i].append(j)
                domination_counts[j] += 1
    # 初始化第一前沿面（Front 1）
    fronts[0] = [i for i in range(num_features) if domination_counts[i] == 0]
    # 构建后续前沿面
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_features[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
    # 3. 返回所有前沿（每个前沿内的特征按原始索引升序排列）
    return [sorted(front) for front in fronts]


def train_and_test(model, x_train, x_test, y_train, y_test):
    scaler = StandardScaler()  # 数据的标准化
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model.fit(x_train, y_train)  # 模型训练
    y_test_pred_proba = model.predict_proba(x_test)
    gmean, mauc, f1 = calculate_gmean_mauc_f1(y_test_pred_proba, y_test)  # 计算准确率指标
    return (gmean, mauc, f1)