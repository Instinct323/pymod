import random
from typing import Type

import numpy as np
import pandas as pd
from tqdm import trange


class ChromosomeBase:
    ''' 染色体基类'''

    def __init__(self, *args, **kwargs):
        ''' 无参构造: 用于生成随机个体
            有参构造: 用于遗传交叉、基因突变'''
        raise NotImplementedError

    def __eq__(self, other):
        ''' 重载 == 运算符, 用于去重'''
        raise NotImplementedError

    def fitness(self) -> float:
        ''' 适应度函数 (max -> best)'''
        raise NotImplementedError

    def variation(self):
        ''' 基因突变'''
        raise NotImplementedError

    def cross_with(self, other):
        ''' 交叉遗传'''
        raise NotImplementedError


class GeneticOpt:
    ''' 遗传算法
        :param chromosome: 染色体类
        :param n_unit: 染色体群体规模
        :param cross_proba: 交叉概率
        :param var_proba: 变异概率
        :param well_radio: 最优个体比例
        :ivar group: 染色体群体'''

    def __init__(self,
                 chromosome: Type[ChromosomeBase],
                 n_unit: int,
                 cross_proba: float = 0.4,
                 var_proba: float = 0.3,
                 well_radio: float = 0.2,
                 best_unit: ChromosomeBase = None):
        self.chromosome = chromosome
        self.n_unit = n_unit
        self.group = self.new_unit(self.n_unit)
        self.log = []

        assert 0 <= cross_proba <= 1, 'cross_proba must be in [0, 1]'
        assert 0 <= var_proba <= 1, 'var_proba must be in [0, 1]'
        self._cross_proba = cross_proba
        self._var_proba = var_proba
        assert cross_proba + var_proba > 0, 'cross_proba + var_proba must be greater than 0'

        assert 0 <= well_radio <= 1, 'well_radio must be in [0, 1]'
        self._well_radio = well_radio

        if isinstance(best_unit, ChromosomeBase): self.group[0] = best_unit

    def new_unit(self, size: int) -> list:
        ''' 初始化染色体群体'''
        return [self.chromosome() for _ in range(size)]

    def sort_unique(self, group: list):
        ''' 计算每个染色体的适应度, 排序、去重'''
        fitness = np.array([x.fitness() for x in group])
        # 去除无限值
        cond = np.isfinite(fitness)
        group, fitness = np.array(group)[cond], fitness[cond]
        # 去除重叠染色体
        order = np.argsort(fitness)[::-1]
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if fitness[order[i]] == fitness[order[j]] and group[order[i]] == group[order[j]]:
                    order[i] = -1
                    break
        # 应用排序结果
        order = order[order != -1]
        return group[order].tolist(), fitness[order]

    def fit(self, epochs: int,
            patience: int = np.inf,
            prefix: str = 'GA-fit') -> np.ndarray:
        ''' :param epochs: 训练轮次
            :param patience: 允许搜索无进展的次数'''
        pbar = trange(epochs)
        angry = 0
        # 最优个体数, 随机选取数
        n_well = max(2, round(self.n_unit * self._well_radio))
        n_choose = self.n_unit - n_well
        # 轮盘赌法采样
        f_sample = lambda p: random.choices(self.group, weights=p)[0]
        for _ in pbar:
            self.group, fitness = self.sort_unique(self.group)
            # 收敛检测
            if self.log and np.isfinite(self.log[-1][0]):
                angry = 0 if fitness[0] > self.log[-1][0] else angry + 1
                if angry == patience: break
            self.log.append([fitness[0], fitness.mean(), fitness.std(), len(fitness)])
            # 保留一定数量的个体
            tmp_group = self.group[:n_well]
            pbar.set_description((f'%-10s' + '%-10.4g') % (prefix, fitness[0]))
            # 使用轮盘赌法进行筛选
            p = fitness - fitness.min()
            p = p / p.sum()
            for _ in range(n_choose):
                pc, pv = np.random.random(2)
                unit = f_sample(p)
                # 交叉遗传 / 基因突变
                if pc <= self._cross_proba: unit = unit.cross_with(f_sample(p))
                if pv <= self._var_proba: unit = unit.variation()
                tmp_group.append(unit)
            self.group = tmp_group
        pbar.close()
        return self.group[0], pd.DataFrame(self.log, columns=['fit-best', 'fit-mean', 'fit-std', 'n-unique'])


if __name__ == '__main__':
    from mod.zjplot import *

    np.random.seed(0)

    N_NODE = 50
    POS = np.random.random([N_NODE, 2]) * 10
    ADJ = np.zeros([N_NODE] * 2, dtype=np.float32)
    # 初始化邻接矩阵
    for i in range(N_NODE):
        for j in range(i + 1, N_NODE):
            ADJ[i][j] = ADJ[j][i] = np.sqrt(((POS[i] - POS[j]) ** 2).sum())


    class Path(ChromosomeBase):
        cluster = None

        @classmethod
        def kmeans(cls):
            from sklearn.cluster import KMeans
            k = N_NODE // 10
            clf = KMeans(n_clusters=k, n_init='auto')
            clf.fit(ADJ)
            cls.cluster = [np.where(clf.labels_ == i)[0] for i in range(k)]
            print('Init cluster.')

        def __init__(self, data=None):
            if isinstance(data, np.ndarray):
                self.data = data
            else:
                if self.cluster:
                    np.random.shuffle(self.cluster)
                    for i in range(len(self.cluster)): np.random.shuffle(self.cluster[i])
                    self.data = np.concatenate(self.cluster)
                else:
                    self.data = np.random.permutation(N_NODE)

        def __eq__(self, other):
            return np.all(self.data == other.data)

        def fitness(self) -> float:
            data = np.concatenate([self.data, self.data[:1]])
            return - ADJ[data[:-1], data[1:]].sum()

        def variation(self):
            ''' 基因突变'''
            l = np.random.randint(0, N_NODE - 1)
            r = np.random.randint(l + 1, N_NODE)
            # note: 对数据进行深拷贝, 否则会影响其他个体
            data = self.data.copy()
            np.random.shuffle(data[l: r + 1])
            return __class__(data)

        def cross_with(self, other):
            ''' 交叉遗传'''
            raise NotImplementedError


    Path.kmeans()
    ga = GeneticOpt(Path, 50, cross_proba=0, var_proba=0.6)
    unit, log = ga.fit(5000)

    regionplot(log['fit-best'], log['fit-mean'], log['fit-std'])
    plt.show()
    print(log)

    # 绘制最优路径
    fig = plt.subplot()
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    plt.plot(*POS[unit.data].T, c='deepskyblue')
    plt.scatter(*POS.T, marker='p', c='orange')
    plt.show()
