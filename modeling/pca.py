import numpy as np

from sklearn.preprocessing import StandardScaler


def association(data):
    ''' 相关系数矩阵 (协方差矩阵)
        data: 形如 [n_sample, n_features] 的 array'''
    return np.cov(StandardScaler().fit_transform(data).T)


def principal(asso, thresh=0.80, dims=None):
    ''' 主成分分析
        asso: 形如 [n_features, n_features] 的 array
        thresh: 累积贡献率阈值
        return: 旋转降维矩阵'''
    cha_value, rota = np.linalg.eig(asso)
    max_idx = np.abs(rota).argmax(axis=0)
    sign = np.sign([c[i] for i, c in zip(max_idx, rota.T)])
    # 得到绝对值最大数的符号
    rota *= sign
    index = np.argsort(cha_value)[::-1]
    cha_value, rota = cha_value[index], rota[:, index]
    # 降序排序: 特征值、特征向量
    sum_contribute = np.cumsum(cha_value) / cha_value.sum()
    # 累计贡献率
    if dims: thresh = sum_contribute[dims - 1]
    for idx, sum_con in enumerate(sum_contribute):
        if sum_con >= thresh:
            info_contribute = cha_value[:idx + 1] / cha_value.sum()
            # 取前 m 个主成分, 计算信息贡献率
            # print(f'主成分贡献率: {info_contribute}\n')
            rota = rota[:, :idx + 1]
            return rota
