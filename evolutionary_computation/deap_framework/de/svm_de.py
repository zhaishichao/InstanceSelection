from array import array
import scipy.io as sio  # 从.mat文件中读取数据集
import numpy as np
from deap import base
from deap import creator
from deap import tools
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from deap_framework.de.custom_functions import *
from utils.dataset_utils import get__counts

################################################################加载数据集################################################
# 数据集
mat_data = sio.loadmat('../../../data/dataset/Australian.mat')
# 提取变量
dataset_x = mat_data['X']
dataset_y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
# 显示数据集分布
print("x的形状:", dataset_x.shape)
print("y的形状:", dataset_y.shape)
# 统计每个类别的个数，dataset_y.max()+1是类别的个数
classes, counts= get_classes_indexes_counts(dataset_y)
print("每种类别的数量：", counts)

# 通过管道将标准化操作和模型相连接
model = make_pipeline(StandardScaler(), SVC(kernel='linear', cache_size=600))  # 'linear' 是线性核，也可以选择 'rbf', 'poly' 等核函数
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=42)
def SVM_Error_Rate(x):  # x的维度为10，也即D=10
    error_rate = objective_function(x, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model=model)
    return [error_rate]



IND_DIM = x_train.shape[0]
save_path="C://Users//zsc//Desktop//evolution computation//experiment//imgs"

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", generate_random_numbers, 3, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_DIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", mutDE, f=0.5)
toolbox.register("mate", cxBinomial, cr=0.9)
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", SVM_Error_Rate)

def main():

    RUN = 20
    per_genera_best_individual = []
    per_generation_best_instances_counts=[]
    per_generation_beat_average_fitness=[]
    with tqdm(total=RUN, desc="DE") as pbar:
        for i in range(RUN):
            print(f"######################第{i+1}次迭代开始#########################")
            NUM_POP = 50
            NGEN = 100
            x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3,
                                                                random_state=np.random.randint(RUN))
            get__counts(y_train,True)
            pop = toolbox.population(n=NUM_POP);
            hof = tools.HallOfFame(1)

            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for g in range(1, NGEN):
                children = []
                for agent in pop:
                    # We must clone everything to ensure independence
                    a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
                    x = toolbox.clone(agent)
                    y = toolbox.clone(agent)
                    y = toolbox.mutate(y, a, b, c)
                    z = toolbox.mate(x, y)
                    del z.fitness.values
                    children.append(z)
                fitnesses = toolbox.map(toolbox.evaluate, children)
                for (i, ind), fit in zip(enumerate(children), fitnesses):
                    ind.fitness.values = fit
                    if ind.fitness > pop[i].fitness:
                        pop[i] = ind
                hof.update(pop)

            print("with fitness", hof[0].fitness.values[0])
            pbar.set_postfix({
                "当前迭代次数": i + 1,
                "当前最高准确率": round(1 - hof[0].fitness.values[0], 3)
            })
            # 更新进度条
            pbar.update(1)
            pop_best=np.round(hof[0]).astype(int)
            per_genera_best_individual.append(pop_best)
            get__counts(pop_best)
            classes, counts = get_classes_indexes_counts(y_train)
            x_best_sub, y_best_sub, xi = get_sub_dataset(pop_best, get_indices(pop_best), x_train, y_train, classes,
                                                         2)
            classes_x_best, counts_x_best = get_classes_indexes_counts(y_best_sub)
            print("最优实例子集各分类数量：", counts_x_best)
            per_generation_best_instances_counts.append(str(counts_x_best))
            per_generation_beat_average_fitness.append(round(1 - hof[0].fitness.values[0],4)*100)
    # 设置可显示中文宋体
    plt.rcParams['font.family'] = 'STZhongsong'
    name="Australian2"
    plt.title("DataSet: "+name)
    plt.xlabel("Best Instance Selection")
    plt.ylabel("Average accuracy (%)")
    plt.bar(per_generation_best_instances_counts, per_generation_beat_average_fitness, width=0.8)
    plt.tick_params(axis='x', labelsize=7)
    for a, b in zip(per_generation_best_instances_counts, per_generation_beat_average_fitness):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    plt.savefig(save_path+"//"+name, dpi=300, bbox_inches='tight')
    plt.close()
    per_genera_best_individual = np.array(per_genera_best_individual)
    print(f"平均准确率：{np.mean(per_generation_beat_average_fitness)}")
if __name__ == "__main__":
    main()
