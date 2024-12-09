import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from instance_selection.multi_objective.ensemble_operator import vote_ensembles
from instance_selection.multi_objective.genetic_operator import selNSGA2, mutate_binary_inversion, selTournamentDCD, \
    exponential_distribution, find_duplicates, remove_duplicates, fitness_function
import warnings

from utils.dataset_utils import get_subset, get_classes_indexes_counts

warnings.filterwarnings("ignore")  # 忽略警告
from sklearn.neural_network import MLPClassifier

import array
import random
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from matplotlib.animation import FuncAnimation, PillowWriter


class InstanceSelection_NSGA2:
    def __init__(self, x_train, y_train, x_test, y_test,seed):
        # 初始化数据集
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.seed = seed
        # 统计每个类别的个数
        classes_train, counts_train = get_classes_indexes_counts(y_train)
        # 计算每个类的权重
        self.weights_train = (1 / counts_train.astype(float)) / np.sum(1 / counts_train.astype(float))
        # 最大化评价目标类
        creator.create("FitnessMaxAndMax", base.Fitness, weights=(1.0, 1.0, 1.0))
        # 个体类
        creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMaxAndMax, pfc=None, mlp=None,
                       y_sub_and_pred=None)
        self.toolbox = base.Toolbox()
        # 训练集实例数量
        self.NDIM = len(y_train)
        # 设置参数
        # 指数分布的参数λ（lambda）在下面的函数中，该值越大越偏向于1
        self.lambda_ = 1.3
        # 阈值（阈值决定了生成0或1）
        self.threshold = 1.0

        # 二进制编码
        self.toolbox.register("attr_binary", exponential_distribution, self.lambda_, self.threshold)  # 0-1编码
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_binary,
                              n=self.NDIM)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 适应度函数
        self.toolbox.register("evaluate", fitness_function, weights_train=self.weights_train)

        # 单点交叉
        self.toolbox.register("mate", tools.cxOnePoint)

        # 二进制突变
        self.toolbox.register("mutate", mutate_binary_inversion)

        # NSGA-II选择（非支配排序后）
        self.toolbox.register("select", selNSGA2, x_test=x_test, y_test=y_test)

        # 找到种群中重复个体的索引对
        self.toolbox.register("find_duplicates", find_duplicates)
        # 去重
        self.toolbox.register("remove_duplicates", remove_duplicates)

        # 绘图
        self.fig = plt.figure(figsize=(12.8, 9.6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # 动画中的数据
        self.pareto_fronts_history = []

    # 绘制动态图
    def update(self, frame):
        self.ax.clear()

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.pareto_fronts_history[frame])))

        for i, front in enumerate(self.pareto_fronts_history[frame]):
            front_points = np.array([ind.fitness.values for ind in front])
            self.ax.scatter(front_points[:, 0], front_points[:, 1], front_points[:, 2], color=colors[i],
                            label=f"Front {i}")
        self.ax.set_title(f"Generation {frame}")
        self.ax.set_xlabel("Acc 1")
        self.ax.set_ylabel("Acc 2")
        self.ax.set_zlabel("Acc 3")
        # ticks = np.linspace(0, 1, 10)
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        # ax.set_zticks(ticks)
        self.ax.legend()

    # 算法得执行
    def run(self, hidden_size, max_iter):
        print("#########################开始NSGA-II算法#########################")
        random.seed(self.seed)

        NGEN = 30  # 迭代次数
        POPSIZE = 40  # 种群数量
        CXPB = 1.0  # 交叉因子/交叉率
        MR = 0.25  # 突变因子/突变率
        # MLP
        learning_rate = 0.001
        hidden_size = hidden_size
        max_iter = max_iter
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 配置五折交叉验证

        ####################################迭代过程的记录#############################
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        logbook = tools.Logbook()
        logbook.header = "gen", "fronts", "fronts_0_size", "Acc1", "Acc2", "Acc3", "recall_per_class", "Gmean", "mAUC"
        ####################################种群的初始化###########################
        pop = self.toolbox.population(n=POPSIZE)
        ####################################计算初始种群的适应度###########################
        ensembles = []  # Pareto front中个体的mlp集成
        # 对于每个个体都训练得到一个mlp模型，并计算适应度
        for i in range(len(pop)):
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=max_iter, random_state=self.seed,
                                learning_rate_init=learning_rate)
            x_sub, y_sub = get_subset(pop[i], self.x_train, self.y_train)
            # 使用 cross_val_predict 进行交叉验证并获取预测
            y_pred = cross_val_predict(mlp, x_sub, y_sub, cv=cv)
            # 用实例选择的子集训练模型
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=max_iter, random_state=self.seed,
                                learning_rate_init=learning_rate)
            mlp.fit(x_sub, y_sub)
            pop[i].mlp = mlp
            pop[i].y_sub_and_pred = (y_sub, y_pred)  # 更新当前个体的y_sub_and_pred（训练子集和对应的预测结果）
            pop[i].fitness.values = self.toolbox.evaluate(pop[i])  # 由mlp模型得到个体的适应度
        #################################计算PFC并进行非支配排序#########################################
        # 计算PFC并进行非支配排序 PFC代替拥挤距离
        pop, pareto_fronts = self.toolbox.select(pop, len(pop))
        ####################################种群的迭代#################################################
        for gen in range(1, NGEN + 1):
            print(f'第{gen}代开始')
            # 选择
            offspring = selTournamentDCD(pop, POPSIZE)
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            # 交叉
            for i in range(0, len(offspring) - 1, 2):
                if random.random() <= CXPB:
                    offspring[i], offspring[i + 1] = self.toolbox.mate(offspring[i], offspring[i + 1])
                # 突变
                offspring[i] = self.toolbox.mutate(offspring[i], MR)[0]
                offspring[i + 1] = self.toolbox.mutate(offspring[i + 1], MR)[0]
                del offspring[i].fitness.values, offspring[i + 1].fitness.values
            # 计算新的种群适应度
            for i in range(len(offspring)):
                mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=max_iter, random_state=self.seed,
                                    learning_rate_init=learning_rate)
                x_sub, y_sub = get_subset(offspring[i], self.x_train, self.y_train)
                # 使用 cross_val_predict 进行交叉验证并获取预测
                y_pred = cross_val_predict(mlp, x_sub, y_sub, cv=cv)
                # 用实例选择的子集训练模型
                mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=max_iter, random_state=self.seed,
                                    learning_rate_init=learning_rate)
                mlp.fit(x_sub, y_sub)
                offspring[i].mlp = mlp
                offspring[i].y_sub_and_pred = (y_sub, y_pred)
                offspring[i].fitness.values = self.toolbox.evaluate(offspring[i])
            # 种群的合并
            pop = pop + offspring
            ###############################################得到pareto_fronts############################################
            pop, pareto_fronts = self.toolbox.select(pop, POPSIZE)
            # pop, pareto_fronts = toolbox.select(pop, POPSIZE)
            record = stats.compile(pop)
            # 保存第一个等级里的mlp模型进行集成
            for ind in pareto_fronts[0]:
                ensembles.clear()
                ensembles.append(ind.mlp)
            # 显示集成个体中的三个评价指标
            Acc1_list = []
            Acc2_list = []
            Acc3_list = []
            for ind in pareto_fronts[0]:
                Acc1_list.append(ind.fitness.values[0])
                Acc2_list.append(ind.fitness.values[1])
                Acc3_list.append(ind.fitness.values[2])
            # 求Acc1_list、Acc2_list、Acc3_list的平均值
            Acc1_mean = round(sum(Acc1_list) / len(Acc1_list), 4)
            Acc2_mean = round(sum(Acc2_list) / len(Acc2_list), 4)
            Acc3_mean = round(sum(Acc3_list) / len(Acc3_list), 4)

            g_mean, m_auc, recall_per_class = vote_ensembles(ensembles, self.x_test, self.y_test)
            logbook.record(gen=gen, fronts=len(pareto_fronts), fronts_0_size=len(pareto_fronts[0]),
                           Acc1=Acc1_mean, Acc2=Acc2_mean, Acc3=Acc3_mean, recall_per_class=recall_per_class,
                           Gmean=g_mean,
                           mAUC=m_auc,
                           **record)
            print(logbook.stream)
            # 清除并绘制当前一代的 Pareto-front
            self.pareto_fronts_history.append(pareto_fronts)

        # 使用 FuncAnimation 生成动画
        savepath = "C:/Users/sc_zh/Desktop/"
        writer = PillowWriter(fps=10)  # 设置帧率
        anim = FuncAnimation(self.fig, self.update, frames=len(self.pareto_fronts_history), interval=200)
        anim.save(savepath + "datasetname" + "_pareto_front.gif", writer=writer)
        plt.show()
        return pop, stats, ensembles
