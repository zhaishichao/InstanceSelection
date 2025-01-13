import numpy as np
from scipy.stats import gmean
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from instance_selection.operator.metrics import calculate_gmean_mauc
from instance_selection.parameter.parameter import *
import scipy.io as sio  # 从.mat文件中读取数据集



class SAMME:
    """
    SAMME - multi-class AdaBoost algorithm
    @ref:   Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. Statistics and its
            interface. 2. 10.4310/SII.2009.v2.n3.a8.
    """

    def __init__(self, num_learner: int, base_classifier, X: np.ndarray, y: np.ndarray):
        """
        Constructor
        :param num_learner: number of weak learners that will be boosted together
        :param num_cats: number of categories
        """
        self.num_cats = np.unique(y).size
        if self.num_cats < 2:
            raise Exception("Param num_cat should be at least 2 but was {}".format(self.num_cats))

        self.num_learner = num_learner
        self.num_cats = np.unique(y).size
        self.entry_weights = None
        self.learner_weights = None
        self.sorted_learners = None
        list_of_learner = []
        for i in range(num_learner):
            classifier = clone(base_classifier)
            # classifier.random_state = classifier.random_state + i
            classifier.random_state = classifier.random_state + np.random.randint(0, 1000)
            classifier.fit(X, y)
            list_of_learner.append(classifier)
        self.learners = list_of_learner
        # 将X,y中的每一条数据构成一个元组
        self.train_data = [(x, y) for x, y in zip(X, y)]
        self.x = X
        self.y = y

    def train(self):
        """
        Train the AdaBoost .
        The training data need to be in the format: [(X, label), ...]
        The learners need to be in the format: [obj1, obj2, ...]
        The learner object need to have: a predict method that can output the predicted class. obj.predict(X) -> cat: int
        :param train_data: training data
        :param learners: weak learners
        :return: void
        """

        # print("\nStart training SAMME..")
        # initialize the weights for each data entry
        n, m = len(self.train_data), len(self.learners)
        self.entry_weights = np.full((n,), fill_value=1 / n, dtype=np.float32)
        self.learner_weights = np.zeros((m,), dtype=np.float32)

        # sort the weak learners by error
        error = [0 for i in range(m)]
        for learner_idx, learner in enumerate(self.learners):
            y_predicted = learner.predict(self.x)
            # 统计y_predicted和self.y中预测错误的数量
            error[learner_idx] = sum(
                1 for label, predicted_label in zip(self.y, y_predicted) if label != predicted_label)
        # for learner_idx, learner in enumerate(self.learners):
        #     for entry in self.train_data:
        #         X, label = entry[0], int(entry[1])
        #         predicted_cat = learner.predict(X)
        #         if predicted_cat != label:
        #             error[learner_idx] += 1
        self.sorted_learners = [l for l, e in sorted(zip(self.learners, error), key=lambda pair: pair[1])]

        # boost
        for learner_idx, learner in enumerate(self.sorted_learners):
            # compute weighted error
            # is_wrong = np.zeros((n,))
            y_predicted = learner.predict(self.x)
            is_wrong = np.not_equal(self.y, y_predicted).astype(int)

            # for entry_idx, entry in enumerate(self.train_data):
            #     X, label = entry[0], int(entry[1])
            #     predicted_cat = learner.predict(X)
            #     if predicted_cat != label:
            #         is_wrong[entry_idx] = 1
            weighted_learner_error = np.sum(is_wrong * self.entry_weights) / self.entry_weights.sum()

            # compute alpha, if the learner is not qualified, set to 0
            self.learner_weights[learner_idx] = max(0, np.log(1 / (weighted_learner_error + 1e-6) - 1) + np.log(
                self.num_cats - 1))
            alpha_arr = np.full((n,), fill_value=self.learner_weights[learner_idx], dtype=np.float32)
            # update entry weights, prediction made by unqualified learner will not update the entry weights
            self.entry_weights = self.entry_weights * np.exp(alpha_arr * is_wrong)
            self.entry_weights = self.entry_weights / self.entry_weights.sum()

        # normalize the learner weights
        self.learner_weights = self.learner_weights / self.learner_weights.sum()
        # print("Training completed.")

    def predict(self, X):
        """
        Predict using the boosted learner
        :param X:
        :return: predict class
        """

        # 获取样本数量
        num_samples = X.shape[0]
        # 初始化预测池，维度为 (num_samples, num_cats)
        pooled_prediction = np.zeros((num_samples, self.num_cats), dtype=np.float32)

        # 遍历所有弱学习器
        for learner_idx, learner in enumerate(self.sorted_learners):
            # 当前弱学习器对所有样本的预测
            predicted_cats = learner.predict(X)  # 假设 predict 支持批量预测，返回形状为 (num_samples,)

            # 初始化平衡预测数组，维度为 (num_samples, num_cats)
            prediction = np.full((num_samples, self.num_cats),
                                 fill_value=-1 / (self.num_cats - 1), dtype=np.float32)

            # 将对应预测类别的位置设置为 1
            prediction[np.arange(num_samples), predicted_cats] = 1

            # 加权累加到预测池
            pooled_prediction += prediction * self.learner_weights[learner_idx]

        # 返回每个样本的最终预测类别
        return np.argmax(pooled_prediction, axis=1)
    def predict_prob_pseudo(self, X):
        """
        Predict using the boosted learner
        :param X:
        :return: predict class
        """

        # 获取样本数量
        num_samples = X.shape[0]
        # 初始化预测池，维度为 (num_samples, num_cats)
        pooled_prediction = np.zeros((num_samples, self.num_cats), dtype=np.float32)

        # 遍历所有弱学习器
        for learner_idx, learner in enumerate(self.sorted_learners):
            # 当前弱学习器对所有样本的预测
            predicted_cats = learner.predict(X)  # 假设 predict 支持批量预测，返回形状为 (num_samples,)

            # 初始化平衡预测数组，维度为 (num_samples, num_cats)
            prediction = np.full((num_samples, self.num_cats),
                                 fill_value=-1 / (self.num_cats - 1), dtype=np.float32)

            # 将对应预测类别的位置设置为 1
            prediction[np.arange(num_samples), predicted_cats] = 1

            # 加权累加到预测池
            pooled_prediction += prediction * self.learner_weights[learner_idx]
        # 对每一行进行softmax归一化
        exp_preds = np.exp(pooled_prediction - np.max(pooled_prediction, axis=1, keepdims=True))  # 防止溢出
        softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        # 返回每个样本的最终预测类别
        return softmax_preds
    def predict_prob(self, X):
        """
        Predict using the boosted learner
        :param X:
        :return: predict class
        """
        def _samme_proba(estimator, n_classes, X):
            proba = estimator.predict_proba(X)
            np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
            log_proba = np.log(proba)
            return (n_classes - 1) * (log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

        pred = sum(
            _samme_proba(estimator, len(getattr(estimator, "classes_", None)), X) for estimator in self.sorted_learners
        )
        pred /= self.learner_weights.sum()
        # 对每一行进行softmax归一化
        exp_preds = np.exp(pred - np.max(pred, axis=1, keepdims=True))  # 防止溢出
        softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        return softmax_preds


if __name__ == "__main__":
    DATASET = Satellite  # 数据集名称（包含对应参数的字典形式）
    datasetname = DATASET['DATASETNAME'].split('.')[0]
    mat_data = sio.loadmat(IMBALANCED_DATASET_PATH + DATASET['DATASETNAME'])  # 加载、划分数据集
    x = mat_data['X']
    y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_SEED)  # 划分数据集
    scaler = StandardScaler()  # 数据的标准化
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = MLPClassifier(hidden_layer_sizes=(DATASET['HIDDEN_SIZE'],), max_iter=DATASET['MAX_ITER'],
                          random_state=RANDOM_SEED, learning_rate_init=DATASET['LEARNING_RATE'])
    gmean_results = []
    mauc_results = []
    num_runs = 30
    for i in range(num_runs):
        samme = SAMME(POPSIZE, model, x_train, y_train)
        samme.train()
        y_pred_prob = samme.predict_prob(x_test)
        gmean, mauc, recall_per_class = calculate_gmean_mauc(y_pred_prob, y_test)

        print(f"第{i + 1}次执行，G-Mean: {gmean},mAUC: {mauc},Recall: {recall_per_class}")
        gmean_results.append(gmean)
        mauc_results.append(mauc)
    print(f"Avg-G-Mean: {np.mean(gmean_results, axis=0)}")
    print(f"Avg-mAUC: {np.mean(mauc_results, axis=0)}")
