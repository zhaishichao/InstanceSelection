# 引用自
# [E. R. Q. Fernandes, A. C. P. L. F. de Carvalho and X. Yao
# “Ensemble of Classifiers Based on Multiobjective Genetic Sampling for Imbalanced Data,”
# in IEEE Transactions on Knowledge and Data Engineering, vol. 32, no. 6, pp. 1104-1115, 1 June 2020
# doi: 10.1109/TKDE.2019.2898861.]

# 1、得到平衡数据集
########################################################################################################################
import numpy as np

from instance_selection.multi_objective.ensemble_operator import vote_ensembles
from instance_selection.multi_objective.backup.instance_selection_base_nsga2 import InstanceSelection_NSGA2
from utils.dataset_utils import get_classes_indexes_counts, balanced_dataset
import scipy.io as sio  # 从.mat文件中读取数据集

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 随机种子
random_seed = 42
print("#########################加载数据集#########################")
# 数据集
# Nursery(0.1)、Satellite(0.001)、Contraceptive(0.1)
datasetname = 'Satellite.mat'
mat_data = sio.loadmat('../../data/dataset/' + datasetname)

dataset_x = mat_data['X']
dataset_y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
# 显示数据集分布
print("特征数据:", dataset_x.shape)
print("label:", dataset_y.shape)
# 统计每个类别的个数
classes, counts = get_classes_indexes_counts(
    dataset_y)  # np.argmax(y_onehot, axis=1)找最大值的索引，将0-1序列转化为0,1,2,3......的整数标签
print("每种类别的分布：", counts)
print("#########################划分数据集#########################")
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=random_seed)

# 数据的标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 显示数据集分布
print("特征数据:", x_train.shape)
print("label:", y_train.shape)
# 统计每个类别的个数
classes_train, counts_train = get_classes_indexes_counts(y_train)
# 计算每个类的权重
weights_train = (1 / counts_train.astype(float)) / np.sum(1 / counts_train.astype(float))
print("训练集每种类别的分布：", counts_train)
print("训练集每种类别的权重：", weights_train)
classes_test, counts_test = get_classes_indexes_counts(y_test)
print("测试集每种类别的分布：", counts_test)
print("#########################平衡数据集#########################")
# 确定每个类别的分布
num_instances = int(counts_train.min() * 1.0)  # 向下取整
num_instances_train = len(y_train)  # 取训练集的数量
print("最小数量:", num_instances)
# 在每个类别中随机的选择该数量的实例的索引
balanced_dataset_x, balanced_dataset_y = balanced_dataset(x_train, y_train, num_instances)
balanced_dataset_x = np.array(balanced_dataset_x)
balanced_dataset_y = np.array(balanced_dataset_y).astype(int)
# 显示数据集分布
print("平衡的数据集的特征数据:", balanced_dataset_x.shape)
print("label:", balanced_dataset_y.shape)
# 统计每个类别的分布
classes_balanced_dataset, counts_balanced_dataset = get_classes_indexes_counts(balanced_dataset_y)
print("平衡的数据集中每种类别的分布：", counts_balanced_dataset)

if __name__ == "__main__":

    print("#########################进行实例选择#########################")
    instance_selection_nsga2 = InstanceSelection_NSGA2(x_train, y_train, x_test, y_test, seed=random_seed)
    hidden_size = 20
    max_iter = 1000
    pop, stats, ensembles = instance_selection_nsga2.run(hidden_size, max_iter)
    print("##############################集成分类器的预测结果：################################")
    g_mean, m_auc, recall_per_class = vote_ensembles(ensembles, x_test, y_test, show_result=True)
    print(f"最终的集成分类结果：Recall_Per_Class{recall_per_class}，Gmean：{g_mean}，mAUC：{m_auc}")
