import pandas as pd
import scipy.io as sio  # 从.mat文件中读取数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metrics import calculate_gmean_mauc

IMBALANCED_DATASET_PATH = 'D:/Develop/WorkSpace/Python/InstanceSelection/datasets/mat/imbalance/'


class EnsembleTest():
    def __init__(self, datasets, num_run):
        self.datasets = datasets
        self.num_run = num_run
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def pre_process(self, dataset, random_state):
        mat_data = sio.loadmat(IMBALANCED_DATASET_PATH + dataset.DATASETNAME)  # 加载、划分数据集
        x = mat_data['X']
        y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                            random_state=random_state)  # 划分数据集
        scaler = StandardScaler()  # 数据的标准化
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_test(self, model, save_path):
        gmean_list = []
        mauc_list = []
        df_gmean = pd.DataFrame()
        df_mauc = pd.DataFrame()
        for dataset in self.datasets:
            for i in range(self.num_run):
                self.pre_process(dataset, i)
                model.fit(self.x_train, self.y_train)
                y_pred_proba = model.predict_proba(self.x_test)  # 默认预测结果是软标签
                gmean, mauc = calculate_gmean_mauc(y_pred_proba, self.y_test)
                gmean_list.append(gmean)
                mauc_list.append(mauc)
            df_gmean[dataset.DATASETNAME.split('.')[0]] = gmean_list
            df_mauc[dataset.DATASETNAME.split('.')[0]] = mauc_list
        df_gmean.to_csv(save_path + 'gmean.csv')
        df_mauc.to_csv(save_path + 'mauc.csv')