from skfeature.function.information_theoretical_based import MIFS,FCBF
from sklearn.base import clone
from config import Datasets
from feature_selection import FeatureSelection, train_and_test
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

import warnings

# warnings.filterwarnings("ignore")  # 忽略警告

fs = FeatureSelection(Datasets)

# 数据预处理
fs.pre_process(Datasets[0], random_state=42)
fs.display_distribution()
# 前后结果对比（原始数据、SMOTE、特征选择+SMOTE）
model = MLPClassifier(hidden_layer_sizes=(fs.dataset.HIDDEN_SIZE,), max_iter=fs.dataset.MAX_ITER,
                      random_state=42, learning_rate_init=fs.dataset.LEARNING_RATE)


res_1 = train_and_test(clone(model), fs.x_train, fs.x_test, fs.y_train, fs.y_test)
print(f"原始：{res_1}")
x_train, y_train = SMOTE(random_state=42, k_neighbors=fs.dataset.K_NEIGHBORS).fit_resample(fs.x_train, fs.y_train)
res_2 = train_and_test(clone(model), x_train, fs.x_test, y_train, fs.y_test)
print(f"SMOTE：{res_2}")

#idx_3 = fs.feature_selection(FCBF.fcbf, mode='index', n_selected_features=fs.x_train.shape[1])
idx_4 = fs.feature_selection(MIFS.mifs, mode='index', n_selected_features=fs.x_train.shape[1])

#print(idx_3)
print(idx_4)