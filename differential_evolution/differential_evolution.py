import numpy as np
import random


class DifferentialEvolution:
    Threshold = 1e-6

    def __init__(self, NP, D, G, CR, Threshold, F, Left, Right):
        self.NP = NP  # 个体数目
        self.D = D  # 目标函数中变量的个数
        self.G = G  # 最大迭代数
        self.CR = CR  # 交叉算子
        self.Threshold = Threshold  # 阈值
        self.F = F  # 变异算子
        self.Left = Left  # 左边界
        self.Right = Right  # 右边界

    # 差分变异，发生在不同的个体之间
    def variation(self, x):
        # 初始化变异个体
        v = np.zeros(x.shape)
        for i in range(0, x.shape[0]):

            # 表示的是在0-NP范围内，随机生成3个整数，作为索引，且保证3个索引不与当前的循环次数i重复（通过这种方式来确保至少有一个个体是变异了的）
            randoms = random.sample(range(0, x.shape[0]), 3)  # 表示的是在0-x.shape[1]范围内，随机生成3个整数，作为索引
            while randoms[0] == i or randoms[1] == i or randoms[2] == i:
                randoms = random.sample(range(0, x.shape[0]), 3)
            # 计算变异的个体
            v[i, :] = x[randoms[0], :] + self.F * (x[randoms[1], :] - x[randoms[2], :])
        return v

    # 变异的优化
    def variation_optimize(self, x):
        # 初始化变异个体
        v = np.zeros(x.shape)
        for i in range(0, x.shape[0]):

            # 表示的是在0-NP范围内，随机生成5个整数，作为索引，且保证5个索引不与当前的循环次数i重复（通过这种方式来确保至少有一个个体是变异了的）
            randoms = random.sample(range(0, x.shape[0]), 5)  # 表示的是在0-x.shape[1]范围内，随机生成5个整数，作为索引
            while randoms[0] == i or randoms[1] == i or randoms[2] == i or randoms[3] == i or randoms[4] == i:
                randoms = random.sample(range(0, x.shape[0]), 5)
            # 计算变异的个体
            v[i, :] = x[randoms[0], :] + self.F * (
                    x[randoms[1], :] + x[randoms[2], :] - x[randoms[3], :] - x[randoms[4], :])
        return v

    # 交叉
    def cross(self, x, v):
        u = np.zeros((self.NP, self.D))
        rate = np.random.rand()
        for i in range(0, x.shape[1]):
            if rate <= self.CR or i == rate:
                u[:, i] = v[:, i]
            else:
                u[:, i] = x[:, i]
        return u

    # 边界处理
    def boundary_process(self, x):
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if x[i][j] < self.Left or x[i][j] > self.Right:
                    x[i][j] = random.random() * (self.Right - self.Left) + self.Left
        return x
