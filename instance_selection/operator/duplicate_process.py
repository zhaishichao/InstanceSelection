######################################
# 重复处理（找到种群中互相重复的个体，并去除）#
######################################

def remove_duplicates(pop, penalty_factor=0.0):
    '''
    :param pop: 要操作的种群对象 array.array
    :param penalty_factor: 差异的惩罚因子（即重复度在0-penalty_factor区间内，均视为重复个体）
    :return: 去重后的pop
    '''
    n = len(pop)
    len_ind = len(pop[0])
    duplicates = []  # 用于记录重复对的索引

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # 计算pop[i]、pop[j]之间的汉明距离（两个二进制序列对应元素不相等的个数）
            hamming_distance = sum(x != y for x, y in zip(pop[i], pop[j]))
            if 1.0 * hamming_distance / len_ind <= penalty_factor:
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)
    # 找到所有需要移除的索引
    to_remove = set()  # 只保留后出现的索引
    for duplicate in duplicates:
        to_remove.update(duplicate)  # update是用来更新set集合的
    return [pop[i] for i in range(len(pop)) if i not in to_remove], len(to_remove)
