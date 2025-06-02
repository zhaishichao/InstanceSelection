import scipy.io as sio  # 从.mat文件中读取数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.dataset_utils import get_distribution

DATASET_PATH = 'D:/Develop/WorkSpace/Python/InstanceSelection/datasets/mat/feature_selection/'


class FeatureSelection():
    def __init__(self, datasets, num_run=None):
        self.datasets = datasets
        self.num_run = num_run
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def pre_process(self, dataset, random_state, standard=False):
        mat_data = sio.loadmat(DATASET_PATH + dataset['DATASETNAME'])  # 加载、划分数据集
        x = mat_data['X']
        y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
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

    def display_distribution(self):
        unique_elements_train, classes_train, counts_train = get_distribution(self.y_train)  # 获取训练集分布
        unique_elements_test, classes_test, counts_test = get_distribution(self.y_test)  # 获取测试集分布
        print(f'trainset distribution: {counts_train}')
        print(f'testset distribution: {counts_test}')
        print(f'number of feature: {self.x_train.shape[1]}')
