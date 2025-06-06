# 数据集路径
IMBALANCED_DATASET_PATH = '../../datasets/mat/imbalance/'
DATASET_PATH = '../../datasets/mat/'
# MLP参数设置
N_SPLITS = 5  # k-fold交叉验证的折数
LEARNING_RATE = 0.1  # 学习率
RANDOM_SEED = 42+1

# 遗传算法迭代参数设置
NGEN = 30  # 迭代次数
POPSIZE = 30  # 种群数量
CXPB = 1.0  # 交叉因子/交叉率
MR = 0.2  # 突变因子/突变率
STOP_SIGN = 5  # 停止标志

# 设置一个类来保存数据集的配置信息
class DatasetConfig:
    def __init__(self, dataset_name, hidden_size, max_iter, learning_rate):
        self.DATASETNAME = dataset_name
        self.HIDDEN_SIZE = hidden_size
        self.MAX_ITER = max_iter
        self.LEARNING_RATE = learning_rate

# 定义数据集配置
Satellite = DatasetConfig('Satellite.mat', 15, 100, 0.1)
Nursery = DatasetConfig('Nursery.mat', 20, 100, 0.1)
Contraceptive = DatasetConfig('Contraceptive.mat', 15, 200, 0.1)
WallRobot = DatasetConfig('WallRobot.mat', 20, 200, 0.1)
Car = DatasetConfig('Car.mat', 20, 200, 0.1)
Balance_Scale = DatasetConfig('Balance_Scale.mat', 15, 500, 0.1)
Ecoli = DatasetConfig('Ecoli.mat', 5, 200, 0.1)
Splice = DatasetConfig('Splice.mat', 5, 100, 0.1)
Glass = DatasetConfig('Glass.mat', 10, 2000, 0.1)
Glass_Init = DatasetConfig('Glass_Init.mat', 10, 2000, 0.1)
Dermatology = DatasetConfig('Dermatology.mat', 2, 1000, 0.1)
Page_Blocks = DatasetConfig('Page_Blocks.mat', 20, 100, 0.1)
Pen_Digits = DatasetConfig('Pen_Digits.mat', 20, 100, 0.1)
# Abalone = DatasetConfig('Abalone.mat', 20, 500, 0.1)
Chess = DatasetConfig('Chess.mat', 20, 200, 0.1)
Chess2 = DatasetConfig('Chess2.mat', 20, 200, 0.1)
# Letter = DatasetConfig('Fars.mat', 10, 1000, 0.1)
German = DatasetConfig('German.mat', 20, 500, 0.1)
Yeast = DatasetConfig('Yeast.mat', 20, 500, 0.1)
USPS = DatasetConfig('USPS.mat', 20, 300, 0.1)
# GuesterPhase = DatasetConfig('GuesterPhase.mat', 50, 400, 0.1)
# Connect = DatasetConfig('Connect.mat', 20, 400, 0.1)
Hayes_Roth = DatasetConfig('Hayes_Roth.mat', 20, 100, 0.1)
Wine = DatasetConfig('Wine.mat', 20, 100, 0.1)
Penbased = DatasetConfig('Penbased.mat', 15, 100, 0.1)
LedDisplay = DatasetConfig('LedDisplay.mat', 10, 100, 0.1)
Shuttle = DatasetConfig('Shuttle.mat', 10, 100, 0.1)
# Lymphography = DatasetConfig('Lymphography.mat', 10, 100, 0.1)
SRBCT = DatasetConfig('SRBCT.mat', 5, 50, 0.1)
Ovarian = DatasetConfig('Ovarian.mat', 10, 100, 0.1)
Automobile = DatasetConfig('Automobile.mat', 15, 200, 0.1)
Led7digit = DatasetConfig('Led7digit.mat', 20, 200, 0.1)
# Zoo = DatasetConfig('Zoo.mat', 5, 200, 0.1)
Satimage = DatasetConfig('Satimage.mat', 15, 200, 0.1)
Lung_Cancer = DatasetConfig('Lung_Cancer.mat', 20, 200, 0.1)
RELATHE = DatasetConfig('RELATHE.mat', 15, 200, 0.1)
# Tae = DatasetConfig('Tae.mat', 5, 200, 0.1)
# Flare = DatasetConfig('Flare.mat', 20, 100, 0.1)
# Kr_vs_k = DatasetConfig('Kr_vs_k.mat', 20, 200, 0.1)
# Cleveland = DatasetConfig('Cleveland.mat', 25, 200, 0.1)