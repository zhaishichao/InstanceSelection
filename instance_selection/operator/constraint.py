from operator import attrgetter


######################################
# 添加约束条件，得到种群中的不可行解和可行解 #
######################################

# 计算cv值
def cv(individual, constraints):
    ind_fitness = individual.fitness.values
    if len(ind_fitness) != len(constraints):
        raise ValueError("约束条件和个体适应度无法匹配！")
    difference = [max(0, x - y) for x, y in zip(constraints, ind_fitness)]  # 判断个体适应度是否都大于等于约束条件
    cv = sum(difference)  # 求0和cv中的最大值之和，cv=0，表示是一个可行解
    individual.fitness.cv = cv  # 将cv值保存在个体中，用于排序
    return cv


# 获得可行解与不可行解
def get_feasible_infeasible(pop, constraints):
    index = []
    for i in range(len(pop)):
        if cv(pop[i], constraints) > 0:  # 判断个体适应度是否都大于等于约束条件
            index.append(i)  # 将不符合约束条件的个体的索引添加到index中
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 得到可行解
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 得到不可行解
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"))  # 对种群中的不可行解按照个体的cv值升序排序
    return feasible_pop, infeasible_pop
