class MILEConfig:
    '''
    MILE算法参数
    '''

    def __init__(self, ngen=30, popsize=30, cxpb=1.0, mr=0.2):
        self.NGEN = ngen  # 迭代次数
        self.POPSIZE = popsize  # 种群大小
        self.CXPB = cxpb  # 交叉概率
        self.MR = mr  # 变异概率


class DataSetConfig:
    '''
    数据集参数
    '''

    def __init__(self, dataset_name, hidden_size, max_iter, learning_rate, k_neighbors):
        self.DATASETNAME = dataset_name  # 数据集名称
        self.HIDDEN_SIZE = hidden_size  # 隐藏层大小
        self.MAX_ITER = max_iter  # 最大迭代次数
        self.LEARNING_RATE = learning_rate  # 学习率
        self.K_NEIGHBORS = k_neighbors


# 数据集配置
GLIOMA = DataSetConfig('GLIOMA.mat', 5, 200, 0.001, 4)
Lung = DataSetConfig('Lung.mat', 10, 100, 0.001, 3)
Ovarian = DataSetConfig('Ovarian.mat', 10, 150, 0.001, 3)
Semeion = DataSetConfig('Semeion.mat', 10, 150, 0.001, 5)
LSVT = DataSetConfig('LSVT.mat', 15, 300, 0.001, 5)
Armstrong_2002_v1 = DataSetConfig('Armstrong-2002-v1.mat', 15, 100, 0.001, 5)
Gordon_2002 = DataSetConfig('Gordon-2002.mat', 15, 150, 0.001, 5)
Colon = DataSetConfig('Colon.mat', 10, 150, 0.001, 5)
Yeoh_2002_v1 = DataSetConfig('Yeoh-2002-v1.mat', 5, 150, 0.001, 5)
DLBCL = DataSetConfig('DLBCL.mat', 10, 150, 0.001, 5)
CNS = DataSetConfig('CNS.mat', 15, 150, 0.001, 5)
Brain2 = DataSetConfig('Brain2.mat', 5, 200, 0.001, 3)
Tumor = DataSetConfig('11Tumor.mat', 20, 200, 0.001, 3)
GLI_85 = DataSetConfig('GLI-85.mat', 15, 200, 0.001, 3)

Datasets = [GLIOMA, Lung, Ovarian, Semeion, LSVT, Armstrong_2002_v1, Gordon_2002, Colon, Yeoh_2002_v1, DLBCL, GLI_85]

Datasets_2 = [Armstrong_2002_v1, Gordon_2002, Colon, Yeoh_2002_v1, DLBCL, CNS,
            Brain2, Tumor, GLI_85]

# Datasets_3 = [CNS, Brain2, Tumor, GLI_85]
Datasets_3 = [GLI_85]



Datasets_test = [GLIOMA, Lung, Ovarian, Semeion, Armstrong_2002_v1, Gordon_2002, Colon, Yeoh_2002_v1, CNS,
            Brain2, Tumor, GLI_85]
