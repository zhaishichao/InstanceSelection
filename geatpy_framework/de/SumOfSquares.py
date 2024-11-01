import numpy as np
import geatpy as ea


class SumOfSquares(ea.Problem):
    def __init__(self):
        name = "SumOfSquares"
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化目标最小最大化标记列表，1：min；-1：max
        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化决策变量类型，0：连续；1：离散
        lb = [-10] * Dim
        ub = [10] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界 1表示边界可取，0表示边界不可取
        ubin = [1] * Dim  # 决策变量上边界 1表示边界可取，0表示边界不可取
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        Vars = pop.Phen  # 得到决策变量矩阵

        pop.ObjV = np.sum(Vars ** 2, axis=1, keepdims=True)


