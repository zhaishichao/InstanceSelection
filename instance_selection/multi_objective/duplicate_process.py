######################################
#    查重（找到种群中互相重复的个体）      #
######################################
# 得到重复个体（基于similar）
def find_duplicates_based_on_similarity(pop, similar=0.9):
    """
    找到重复个体的索引。
    :param arrays: 一个包含 array.array 的列表
    :param threshold: 重复的判断阈值
    :return: 重复对的索引列表
    """
    n = len(pop)
    duplicates = []  # 用于记录重复对的索引

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # 当前两组数组
            a = pop[i]
            b = pop[j]

            # 计算1的个数
            ones_a = sum(a)
            ones_b = sum(b)

            # 如果其中一个数组全是0，不可能满足条件
            if ones_a == 0 or ones_b == 0:
                continue

            # 计算交集中的1的数量
            common_ones = sum(x == 1 & y == 1 for x, y in zip(a, b))

            # 判断是否满足重复的定义
            if (common_ones / ones_a > similar) and (common_ones / ones_b > similar):
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)
    return duplicates


# 得到重复个体（完全一样）
def find_duplicates(pop):
    """
    找到重复个体的索引。
    :param arrays: 一个包含 array.array 的列表
    :param threshold: 重复的判断阈值
    :return: 重复对的索引列表
    """
    n = len(pop)
    duplicates = []  # 用于记录重复对的索引

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # 计算pop[i],pop[j]之间的汉明距离
            hamming_distance = sum(x != y for x, y in zip(pop[i], pop[j]))
            if hamming_distance == 0:
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)
    return duplicates


######################################
#               去重                  #
######################################
# 根据索引对，去除种群中重复的个体
def remove_duplicates(pop, duplicates):
    """
    移除重复的个体。
    :param arrays: 一个包含 array.array 的列表
    :param duplicates: 重复对的索引列表
    :return: 去重后的列表
    """
    # 找到所有需要移除的索引
    to_remove = set()  # 只保留后出现的索引
    for duplicate in duplicates:
        to_remove.update(duplicate)  # update是用来更新set集合的
    # 构造去重后的列表
    return [pop[i] for i in range(len(pop)) if i not in to_remove], len(to_remove)

