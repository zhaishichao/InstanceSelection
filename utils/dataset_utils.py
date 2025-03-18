from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
import numpy as np


# 删除指定类别的数据
def remove_class(X, y, class_to_remove):
    """
    删除指定类别的数据

    :param X: ndarray, 特征数据, shape (n_samples, n_features)
    :param y: ndarray, 标签数据, shape (n_samples,)
    :param class_to_remove: int, 要删除的类别
    :return: (X_new, y_new) 删除该类别后的特征数据和标签
    """
    mask = y != class_to_remove  # 生成掩码，筛选出不等于 class_to_remove 的样本
    X_new = X[mask]  # 过滤特征
    y_new = y[mask]  # 过滤标签
    return X_new, y_new


def k_fold_cross_validation(model, X, y, n_splits=5, method='soft', random_state=42):
    """
    Perform 5-fold cross-validation and generate soft labels (probability predictions).

    Parameters:
    - model: A sklearn-compatible model with a `predict_proba` method.
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - n_splits:k-fold cross validation
    - method: 'soft' or 'hard'

    Returns:
    - soft_labels: A numpy array containing the soft labels for each sample.
    - scores: A list of accuracy scores for each fold.
    """
    # StratifiedKFold
    # StratifiedKFold 是 KFold 的一个变体，专门用于分类问题中的分层抽样
    # 在每一折中，训练集和测试集中的类别分布与原始数据集中的类别分布一致
    # 适用于分类问题，尤其是当类别分布不均衡时
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)  # 5-fold cross-validation
    soft_labels = np.zeros((len(y), len(np.unique(y))))  # Initialize array for soft labels
    # scores = []
    for train_index, test_index in kf.split(X, y):
        # Split datasets into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Clone and fit the model on the training set
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        # Generate soft labels (probability predictions)
        y_proba = model_clone.predict_proba(X_test)
        soft_labels[test_index] = y_proba
        # Evaluate the model
        # y_pred = np.argmax(y_proba, axis=1)  # Convert probabilities to class predictions
        # score = accuracy_score(y_test, y_pred)
        # scores.append(score)
    if method == 'soft':
        return soft_labels
    elif method == 'hard':
        hard_labels = np.argmax(y_proba, axis=1)
        return hard_labels
    else:
        raise ValueError("Invalid method. Choose 'soft' or 'hard'.")


# 获取数据集的分布

def get_distribution(y):
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, counts = np.unique(y, return_counts=True)
    # 构造每个类别的索引列表
    class_indices = []
    for element in unique_elements:
        class_indices.append(np.where(y == element)[0])
    return unique_elements, class_indices, counts


##########################由个体得到选择的实例子集的索引###########################
def get_indices(individual):
    '''
    :param individual: individual（用二进制或0-1范围内的实值进行编码）
    :return: 被选择实例的索引
    '''
    individual = np.round(individual)  # 数据范围在0-1之间，转化成int的同时会舍去小数部分，从而将个体映射到0-1编码
    indices = np.where(individual == 1)  # 1代表选择该实例，返回值是tuple，tuple[0]取元组中的第一个元素
    return indices[0]


###########################获取实例子集############################
def get_subset(individual, dataset_x, dataset_y):
    '''
    :param individual:
    :return: 实例子集
    '''
    indices = get_indices(individual)
    x_sub = dataset_x[indices, :]
    y_sub = dataset_y[indices]
    return x_sub, y_sub
