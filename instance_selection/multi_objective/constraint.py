######################################
# 添加约束条件，得到种群中的不可行解和可行解 #
######################################
# 计算cv值
from operator import attrgetter


def CV(individual, Acc1, Acc2, Acc3):
    acc = (
        Acc1 - individual.fitness.values[0], Acc2 - individual.fitness.values[1], Acc3 - individual.fitness.values[2])
    cv = 0
    for i in range(len(acc)):
        # 求0和cv中的最大值之和
        cv = cv + max(0, acc[i])
        individual.fitness.cv = cv  # 将cv值保存在个体中
    return cv


# 获得可行解与不可行解
def get_feasible_infeasible(pop, Acc1, Acc2, Acc3):
    index = []
    for i in range(len(pop)):
        if CV(pop[i], Acc1, Acc2, Acc3) > 0:
            index.append(i)
    # 获取可行解与不可行解
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 得到可行解
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 得到不可行解
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"))  # 对种群中的不可行解按照个体的cv值升序排序
    return feasible_pop, infeasible_pop