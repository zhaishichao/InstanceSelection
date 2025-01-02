from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np


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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)  # 5-fold cross-validation
    soft_labels = np.zeros((len(y), len(np.unique(y))))  # Initialize array for soft labels
    # scores = []
    for train_index, test_index in kf.split(X):
        # Split data into train and test
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
    class_indices = {element: np.where(y == element)[0] for element in unique_elements}
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
