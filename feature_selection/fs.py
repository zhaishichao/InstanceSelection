import numpy as np
import scipy.io as sio  # 从.mat文件中读取数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.dataset_utils import get_distribution

DATASET_PATH = 'D:/Develop/WorkSpace/Python/InstanceSelection/datasets/mat/feature_selection/'


class FeatureSelection():
    def __init__(self, datasets, num_run=None):
        self.datasets = datasets
        self.num_run = num_run
        self.dataset = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.distribution = None

    def pre_process(self, dataset, random_state, standard=False):
        mat_data = sio.loadmat(DATASET_PATH + dataset.DATASETNAME)  # 加载、划分数据集
        x = mat_data['X']
        y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
        # 对y进行编码，使其标签为从0开始0，1，2，3...
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        _, _, self.distribution = get_distribution(y)  # 获取类分布
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state,
                                                            stratify=y)  # 划分数据集
        if standard:
            scaler = StandardScaler()  # 数据的标准化
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset = dataset

    def display_distribution(self):
        unique_elements_train, classes_train, counts_train = get_distribution(self.y_train)  # 获取训练集分布
        unique_elements_test, classes_test, counts_test = get_distribution(self.y_test)  # 获取测试集分布
        print(f'trainset distribution: {counts_train}')
        print(f'testset distribution: {counts_test}')
        print(f'number of feature: {self.x_train.shape[1]}')

    def feature_selection(self, func, **kwargs):
        index = func(self.x_train, self.y_train, **kwargs)  # 按照特征的优先级排序（index[0]表示最重要的特征）
        index_convert = self.convert_ranking(index)  # 没有顺序，index_convert[0]表示第一个特征的重要性排名（越小越重要）
        return index_convert

    def convert_ranking(self, index):
        n = len(index)
        arr_revers = np.zeros(n, dtype=int)
        for rank in range(n):
            feature_index = index[rank]
            arr_revers[feature_index] = rank  # 假设排名从0开始
        return arr_revers
