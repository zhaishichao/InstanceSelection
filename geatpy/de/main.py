"""MyProblemMain.py"""
import geatpy as ea  # import geatpy
import numpy as np
import scipy.io as sio  # 从.mat文件中读取数据集
from EISToSVM import EISToSVM # 导入自定义问题接口
from utils.dataset_utils import get__counts

"""============================实例化问题对象========================"""

# 加载数据集

# 读取.mat文件
mat_data = sio.loadmat('../../data/dataset/Australian.mat')
# 提取变量
dataset_x = mat_data['X']
dataset_y = mat_data['Y'][:, 0] # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
# 显示变量信息
print("x的形状:", dataset_x.shape)
print("y的形状:", dataset_y.shape)
# 统计每个类别的个数，dataset_y.max()+1是类别的个数
counts = np.zeros(dataset_y.max() + 1)
for i in range(dataset_y.shape[0]):
    counts[dataset_y[i]] += 1
print("每种类别的数量：", counts)

problem = EISToSVM(dataset_x,dataset_y,random_state=42)  # 实例化问题对象
print("训练集的实例数量：", problem.y_train.shape[0])
print("每种类别的数量：",problem.counts)
"""==============================种群设置==========================="""
Encoding = 'RI'  # 编码方式
NIND = 50  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                  problem.borders)  # 创建区域描述器

population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
"""===========================算法参数设置=========================="""
myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 100  # 最大进化代数
myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
myAlgorithm.recOper.XOVR = 0.7  # 设置交叉概率
myAlgorithm.logTras = 5  # 置每隔多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = True  # 设置是否打印输出日志信息
myAlgorithm.drawing = 3  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）

"""==========================调用算法模板进行种群进化==============="""
[BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
BestIndi.save()  # 把最优个体的信息保存到文件中
"""=================================输出结果======================="""
print('评价次数：%s' % myAlgorithm.evalsNum)
print('时间已过%s 秒' % myAlgorithm.passTime)
if BestIndi.sizes != 0:
    print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
else:
    print('没找到可行解。')