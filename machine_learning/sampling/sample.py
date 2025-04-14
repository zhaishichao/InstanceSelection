import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

from instance_selection.operator.metrics import calculate_gmean_mauc
from utils.dataset_utils import get_distribution


def train_and_test(model, x_train, x_test, y_train, y_test, show_distribution=False):
    scaler = StandardScaler()  # 数据的标准化
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if show_distribution:
        unique_elements_train, classes_train, counts_train = get_distribution(y_train)  # 获取训练集分布
        print(f'trainset: {counts_train}')

    model.fit(x_train, y_train)  # 模型训练
    y_test_pred_proba = model.predict_proba(x_test)

    gmean, mauc, _ = calculate_gmean_mauc(y_test_pred_proba, y_test)  # 计算准确率指标
    return gmean, mauc


def sample_dataset(model, x_train, x_test, y_train, y_test, random_seed, method='NOS'):
    if method == 'ROS':
        x_train, y_train = RandomOverSampler(random_state=random_seed).fit_resample(x_train, y_train)
    if method == 'RUS':
        x_train, y_train = RandomUnderSampler(random_state=random_seed).fit_resample(x_train, y_train)
    if method == 'SMOTE':
        x_train, y_train = SMOTE(random_state=random_seed, k_neighbors=7).fit_resample(x_train, y_train)
    if method == 'ADASYN':
        x_train, y_train = ADASYN(random_state=random_seed).fit_resample(x_train, y_train)
    if method == 'BorderlineSMOTE':
        x_train, y_train = BorderlineSMOTE(random_state=random_seed, k_neighbors=5).fit_resample(x_train, y_train)
    if method == 'KMeansSMOTE':
        x_train, y_train = KMeansSMOTE(random_state=random_seed, cluster_balance_threshold=0.02).fit_resample(x_train,
                                                                                                           y_train)
    gmean, mauc = train_and_test(model, x_train, x_test, y_train, y_test)
    return gmean, mauc
