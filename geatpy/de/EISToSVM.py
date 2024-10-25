import numpy as np
import geatpy as ea
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class EISToSVM(ea.Problem):
    dataset_x = np.empty(shape=())
    dataset_y = np.empty(shape=())
    x_train = np.empty(shape=())
    x_test = np.empty(shape=())
    y_train = np.empty(shape=())
    y_test = np.empty(shape=())
    model = make_pipeline(StandardScaler(),
                          SVC(kernel='linear', cache_size=600))  # 'linear' 是线性核，也可以选择 'rbf', 'poly' 等核函数
    classes = np.empty(shape=())
    counts = np.empty(shape=())
    minimum = 0

    def __init__(self, dataset_x, dataset_y, random_state):

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(dataset_x, dataset_y, test_size=0.3,
                                                                                random_state=random_state)
        self.classes, self.counts = self.get_classes_indices_counts()
        self.minimum = np.min(self.counts) // 2  # 每个类别的实例被选择的最小数量


        name = "EISToSVM"
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化目标最小最大化标记列表，1：min；-1：max
        Dim = self.x_train.shape[0]  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化决策变量类型，0：连续；1：离散
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界 1表示边界可取，0表示边界不可取
        ubin = [1] * Dim  # 决策变量上边界 1表示边界可取，0表示边界不可取

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        Vars = pop.Phen  # 得到决策变量矩阵
        fitness = np.empty(Vars.shape[0])  # 记录种群中个体的适应度
        # 计算每个个体的适应度
        for i in range(0, Vars.shape[0]):
            fitness[i] = self.objective_function(Vars[i])

        pop.ObjV = fitness.reshape(-1, 1)

    """
    1、获取训练集的详情：
        classes_indices：二维ndarray，第一维表示每种类别，第二维表示此类别的实例索引。
        counts：二维ndarray，第一维表示每种类别，第二维表示此类别的数量。
    """

    def get_classes_indices_counts(self):
        # 统计每个类别的个数，y.max()+1是类别的个数
        num_class = self.y_train.max() + 1
        counts = np.zeros(num_class, dtype=int)
        classes = []
        for i in range(self.y_train.shape[0]):  # y.shape[0]相当于y的长度
            counts[self.y_train[i]] += 1
        for i in range(num_class):
            # np.where() 返回值是一个tuple数组，np.where(y == i)[0],表示取出该tuple数组的第一个元素，是一个ndarray数组
            classes.append(np.where(self.y_train == i)[0])
        return classes, counts

    """
    2、根据种群个体的实值编码，获取被选择的个体的索引
    """

    def get_indices(self,xi):
        xi = np.round(xi)  # 数据范围在0-1之间，转化成int的同时会舍去小数部分，从而将个体映射到0-1编码
        indices = np.where(xi == 1)  # 1代表选择该实例，返回值是tuple，tuple[0]取元组中的第一个元素
        return indices[0]

    """
    3、由索引得到实例子集（同时保证子集的最小数量）
    """

    def get_sub_dataset(self, xi, indices):
        # 根据索引得到实例子集
        num_class = len(self.classes)
        x_sub = self.x_train[indices, :]
        y_sub = self.y_train[indices]

        counts_sub = np.zeros(num_class, dtype=int)
        for i in range(y_sub.shape[0]):
            counts_sub[y_sub[i]] += 1
        # 遍历子集中各个类别的数量，保证大于最小数量
        for i in range(num_class):
            # 当实例个数小于minimum，随机添加实例达到最小限制
            if counts_sub[i] < self.minimum:
                # 转换成集合进行差运算（& | -，分别是交、并、差） unselected_indices是一个set集合
                unselected_indices_set = set(self.classes[i]) - set(indices)
                # list(unselected_indices)将集合转换成list
                unselected_indices = np.array(list(unselected_indices_set))
                # replace=False表示不允许重复
                random_selecte_indices = np.random.choice(unselected_indices, size=self.minimum - counts_sub[i],
                                                          replace=False)
                # 添加后更改个体xi的参数
                for j in range(0, self.minimum - counts_sub[i]):  # 小于minimum，添加实例时，需要同步更改xi个体的实值大小，由小于0.5，改为大于0.5
                    xi[random_selecte_indices[j]] = np.random.uniform(0.5, 1)  # 生成0.5-1的随机数
                    index = np.searchsorted(indices, random_selecte_indices[j])
                    indices = np.insert(indices, index, random_selecte_indices[j])
                    x_sub = np.insert(x_sub, index, self.x_train[random_selecte_indices[j], :], axis=0)
                    y_sub = np.insert(y_sub, index, self.y_train[random_selecte_indices[j]])
        return x_sub, y_sub, xi

    # 适应度函数/目标函数
    def objective_function(self, xi):  # xi表示种群的个体
        # 先将xi的实值编码四舍五入得到0-1编码，根据编码得到选择的实例索引
        indices = self.get_indices(xi)
        # 根据索引得到训练子集，并保证实例选择的最小数量
        x_sub, y_sub, xi = self.get_sub_dataset(xi, indices)
        # 模型训练
        self.model.fit(x_sub, y_sub)
        y_pred = self.model.predict(self.x_test)
        # 计算准确率
        accuracy = accuracy_score(self.y_test, y_pred)
        # 计算错误率
        error_rate = 1 - accuracy
        return error_rate
