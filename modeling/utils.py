import logging
import time

import numpy as np
import pandas as pd

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NUMBER = int, float


def laida_bound(data):
    ''' 拉以达边界
        data: 单指标向量'''
    std = np.std(data)
    mean = np.mean(data)
    return mean - std * 3, mean + std * 3


def comentropy(origin):
    ''' 熵权法
        proba: 概率
        info: 信息量 -ln(proba)
        entropy: 信息熵 (Σ proba * info) / ln(n)
        redu: 冗余度 1 - entropy'''
    proba = origin / origin.sum(axis=0, keepdims=True)
    info = - np.log(proba, where=proba > 0)
    redu = 1 - (proba * info).sum(axis=0) / np.log(origin.shape[0])
    return redu / redu.sum()


def solve_weight(array):
    ''' 求解特征方程
        return: w, CI'''
    array, n, wlist = array.copy(), array.shape[0], []
    # 算术平均法
    w = (array / array.sum(axis=0)).mean(axis=1, keepdims=True)
    wlist.append(w)
    # 几何平均法
    w = np.prod(array, axis=1, keepdims=True) ** (1 / n)
    wlist.append(w / w.sum())
    # 特征值法
    solution, ws = np.linalg.eig(array)
    index = solution.argmax()
    lambda_ = solution[index].real
    w = ws[:, index].real
    wlist.append((w / w.sum())[:, None])
    # 输出对比结果
    w = np.concatenate(wlist, axis=1)
    LOGGER.info(pd.DataFrame(w, columns=['算术平均', '几何平均', '特征值']))
    LOGGER.info('')
    w = w.mean(axis=1)
    # 计算 CI
    return w, (lambda_ - n) / (n - 1)


def cal_RI(n, epochs=1000, decimals=4):
    if n <= 15:
        return {3: 0.52, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36, 8: 1.41, 9: 1.46,
                10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}[n]
    else:
        mark = time.time()
        LOGGER.info(f'Calulating RI: dim = {n}, epochs={epochs}, decimals={decimals}')
        lambda_sum = 0
        for i in range(epochs):
            array = np.eye(n)
            # 随机构造成对比较矩阵
            for c in range(n):
                for r in range(c):
                    array[c][r] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                    1 / 2, 1 / 3, 1 / 4, 1 / 5,
                                                    1 / 6, 1 / 7, 1 / 8, 1 / 9])
                    array[r][c] = 1 / array[c][r]
            # 求取最大特征值
            solution, weights = np.linalg.eig(array)
            lambda_sum += solution.max().real
        # 最大特征值的均值
        lambda_ = lambda_sum / 1000
        RI = round((lambda_ - n) / (n - 1), decimals)
        LOGGER.info(f'RI = {RI}, time = {round(time.time() - mark, 2)}s')
        return RI


if __name__ == '__main__':
    pass
