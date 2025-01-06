# MLP参数设置
N_SPLITS = 5  # k-fold交叉验证的折数
LEARNING_RATE = 0.1  # 学习率
RANDOM_SEED = 42

# 遗传算法迭代参数设置
NGEN = 30  # 迭代次数
POPSIZE = 30  # 种群数量
CXPB = 1.0  # 交叉因子/交叉率
MR = 0.2  # 突变因子/突变率
STOP_SIGN = 5  # 停止标志
# 各个数据集参数
Satellite = {
    'DATASETNAME': 'Satellite.mat',
    'HIDDEN_SIZE': 15,
    'MAX_ITER': 100,
    'LEARNING_RATE': 0.1
}

Nursery = {
    'DATASETNAME': 'Nursery.mat',
    'HIDDEN_SIZE': 20,
    'MAX_ITER': 100,
    'LEARNING_RATE': 0.1
}

Contraceptive = {
    'DATASETNAME': 'Contraceptive.mat',
    'HIDDEN_SIZE': 15,
    'MAX_ITER': 200,
    'LEARNING_RATE': 0.1
}

WallRobot = {
    'DATASETNAME': 'WallRobot.mat',
    'HIDDEN_SIZE': 20,
    'MAX_ITER': 200,
    'LEARNING_RATE': 0.1
}

Chess = {
    'DATASETNAME': 'Chess.mat',
    'HIDDEN_SIZE': 20,
    'MAX_ITER': 200,
    'LEARNING_RATE': 0.1
}

Australian = {
    'DATASETNAME': 'Australian.mat',
    'HIDDEN_SIZE': 20,
    'MAX_ITER': 1000,
    'LEARNING_RATE': 0.001
}
