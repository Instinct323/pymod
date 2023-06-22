import logging
import time

import numpy as np
import pandas as pd
import sympy

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

num = (int, float)


def dataframe_set(df):
    ser = {key: set(df[key]) for key in df}
    return pd.Series(ser)


def get_color(n_color):
    heat = np.linspace(0, 255, n_color).round().astype(np.uint8)[None]
    return cv2.applyColorMap(heat, cv2.COLORMAP_RAINBOW)[0].tolist()


def make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack([xv, yv], -1)


class Time_Response:
    ''' 时域响应'''
    s, t = sympy.symbols('s, t')

    def __init__(self, fun, doprint=False):
        ''' fun: 返回关于s的传递函数的 function
            doprint: 输出公式'''
        sys = fun(self.s)
        self.f_t = sympy.integrals.inverse_laplace_transform(sys, s=self.s, t=self.t)
        if doprint:
            sympy.pprint(self.f_t)

    def __call__(self, time):
        ''' 使自身可作为函数被调用'''
        response = list(map(lambda i: float(self.f_t.subs({self.t: i})), time))
        return np.array(response)


def vec_unitize(vec):
    ''' 向量单位化'''
    length = (vec ** 2).sum(axis=1) ** 0.5
    return vec / length.reshape(-1, 1)


def normalize(data, axis=None):
    ''' 归一化
        axis=0: 各列归一化
        axis=1: 各行归一化'''
    data = np.array(data)
    min_ = data.min(axis=axis, keepdims=True)
    max_ = data.max(axis=axis, keepdims=True)
    if min_ == max_: min_ = 0
    return (data - min_) / (max_ - min_)


def standardize(data, axis=None):
    ''' 标准化'''
    data = np.array(data)
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + 1e-4)


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
    num = origin.shape[0]
    proba = origin / origin.sum(axis=0).reshape(1, -1)
    info = -np.log(proba, where=proba > 0)
    entropy = (proba * info).sum(axis=0) / np.log(num)
    redu = 1 - entropy
    weight = redu / redu.sum()
    return weight


def solve_weight(array):
    ''' 求解特征方程
        return: w, CI'''
    array = array.copy()
    dim = array.shape[0]
    weight_list = []
    # 算术平均法
    weight = (array / array.sum(axis=0)).mean(axis=1, keepdims=True)
    weight_list.append(weight)
    # 几何平均法
    weight = np.prod(array, axis=1, keepdims=True) ** (1 / dim)
    weight /= weight.sum()
    weight_list.append(weight)
    # 特征值法
    solution, weights = np.linalg.eig(array)
    index = solution.argmax()
    lambda_ = np.real(solution[index])
    weight = np.real(weights[:, index])
    weight /= weight.sum()
    weight_list.append(weight[:, None])
    # 输出对比结果
    weight = np.concatenate(weight_list, axis=1)
    LOGGER.info(pd.DataFrame(weight, columns=['算术平均', '几何平均', '特征值']))
    LOGGER.info('')
    weight = weight.mean(axis=1)
    # 计算 CI
    CI = (lambda_ - dim) / (dim - 1)
    return weight, CI


def cal_RI(dim, epochs=1000, decimals=4):
    if dim <= 15:
        return {3: 0.52, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36, 8: 1.41, 9: 1.46,
                10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}[dim]
    else:
        mark = time.time()
        LOGGER.info(f'Calulating RI: dim = {dim}, epochs={epochs}, decimals={decimals}')
        lambda_sum = 0
        for i in range(epochs):
            array = np.eye(dim)
            # 随机构造成对比较矩阵
            for col in range(dim):
                for row in range(col):
                    array[col][row] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                        1 / 2, 1 / 3, 1 / 4, 1 / 5,
                                                        1 / 6, 1 / 7, 1 / 8, 1 / 9])
                    array[row][col] = 1 / array[col][row]
            # 求取最大特征值
            solution, weights = np.linalg.eig(array)
            lambda_sum += np.real(solution.max())
        # 最大特征值的均值
        lambda_ = lambda_sum / 1000
        RI = round((lambda_ - dim) / (dim - 1), decimals)
        LOGGER.info(f'RI = {RI}, time = {round(time.time() - mark, 2)}s')
        return RI


if __name__ == '__main__':
    pass
