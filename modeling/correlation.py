import numpy as np


def var_analysis(var):
    ''' 方差分析
        var: [n_sample, n_features]'''
    mean = var.mean(axis=0)
    length = var.shape[0]
    # 总平均值
    mean_all = length * mean.sum() / var.size
    # 组间均方
    msa = length * ((mean - mean_all) ** 2).sum() / (var.shape[1] - 1)
    # 组内均方
    mse = ((var - mean) ** 2).sum() / (var.size - 2)
    return msa / mse


def discrete(x, y, eps=1e-4):
    ''' 离散相关性'''
    get_entropy = lambda proba: - (proba * np.log(proba + eps)).sum()
    pool_x, pool_y = list(set(x)), list(set(y))
    # 对离散变量进行编码
    x = np.array(list(map(lambda x: pool_x.index(x), x)))
    y = np.array(list(map(lambda x: pool_y.index(x), y)))
    # 某特征的概率
    proba_x = np.array([(x == i).sum() for i in range(len(pool_x))]) / len(x)
    proba_y = np.array([(y == i).sum() for i in range(len(pool_y))]) / len(y)
    # 某特征的信息熵
    entropy_x, entropy_y = map(get_entropy, [proba_x, proba_y])
    # 用 x 对 y 分类的信息熵
    proba_clf = []
    for j in range(len(pool_y)):
        sample = x[y == j]
        proba = np.array([(sample == i).sum() for i in range(len(pool_x))]) / len(sample)
        proba_clf.append([len(sample), get_entropy(proba)])
    # 信息增益
    gain = entropy_x - sum([w * p for w, p in proba_clf]) / len(x)
    return gain / (entropy_x * entropy_y)


def pearson(var_a, var_b):
    ''' Pearman 相关系数
        适用条件:
            1. 两变量均应由测量得到的连续变量 (定距变量, 定比变量)
            2. 两变量所来自的总体都应是正态分布, 或接近正态的单峰对称分布。
            3. 变量必须是成对的数据
            4. 两变量间为线性关系
        return: 相关系数 r'''
    var_a = var_a[None] - var_a.mean()
    var_b = var_b[:, None] - var_b.mean()
    cov = var_a @ var_b
    std = [(var_a ** 2).sum(), (var_b ** 2).sum()]
    r = cov / np.sqrt(std[0] * std[1])
    return r


def spearman(var_a, var_b):
    ''' Spearman 等级相关系数
        适用条件:
            按大小 / 优劣排位的定序变量
        return: 相关系数 r, [统计量 t]'''
    n = var_a.size
    assert var_b.size == n
    var_a = np.sort(var_a)
    var_b = np.sort(var_b)
    r = 1 - 6 * ((var_a - var_b) ** 2).sum() / (n * (n ** 2 - 1))
    value = {'r': r}
    if n > 20:
        DOF = n - 2
        t = abs(r * (DOF / (1 - r ** 2)) ** 0.5)
        value['t'] = t
    return value


if __name__ == '__main__':
    a = np.array([1, 2, 3, 1, 2])
    b = np.array([1, 2, 2, 1, 1])
    discrete(a, b)
