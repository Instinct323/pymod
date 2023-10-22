import random

import numpy as np
from tqdm import trange

DTYPE = np.float16


class GeneticOpt:
    ''' 遗传算法
        :param n_unit: 染色体群体规模
        :param n_gene: 染色体的基因数
        :param well_radio: 最优个体比例
        :param cross_proba: 交叉概率
        :param var_proba: 变异概率'''

    def __init__(self,
                 n_unit: int,
                 n_gene: int,
                 well_radio: float = 0.05,
                 cross_proba: float = 0.4,
                 var_proba: float = 0.3):
        self.n_unit = n_unit
        self.n_gene = n_gene
        self._well_radio = well_radio
        self._cross_proba = cross_proba
        self._var_proba = var_proba

    def new_unit(self, size: int) -> np.ndarray:
        ''' 初始化染色体群体
            :return: [size, n_gene]'''
        raise NotImplementedError

    def cross(self, unit: np.ndarray, other: np.ndarray) -> np.ndarray:
        ''' 交叉遗传
            :return: [n_gene, ]'''
        raise NotImplementedError

    def variation(self, unit: np.ndarray) -> np.ndarray:
        ''' 基因突变
            :return: [n_gene, ]'''
        gene_idx = np.arange(self.n_gene)
        l = random.choice(gene_idx)
        r = random.choice(gene_idx[l:])
        np.random.shuffle(unit[l: r + 1])
        return unit

    def fitness(self, group: np.ndarray) -> np.ndarray:
        ''' 适应度函数 (max -> best)'''
        raise NotImplementedError

    def fit(self, epochs: int,
            patience: int = np.inf,
            prefix: str = 'GA_fit') -> np.ndarray:
        ''' :param epochs: 训练轮次
            :param patience: 允许搜索无进展的次数'''
        cur_group = self.new_unit(self.n_unit)
        pbar = trange(epochs)
        last_fitness, angry = - np.inf, 0
        # 最优个体数, 随机选取数
        n_well = max(2, round(self.n_unit * self._well_radio))
        n_choose = self.n_unit - n_well
        for _ in pbar:
            cur_group = np.unique(cur_group, axis=0)
            # 计算每个个体的适应度并排序
            fitness = self.fitness(cur_group)
            order = np.argsort(fitness)[::-1]
            cur_group, fitness = cur_group[order], fitness[order]
            # 收敛检测
            angry = 0 if fitness[0] > last_fitness else angry + 1
            last_fitness = fitness[0]
            if angry == patience: break
            # 保留一定数量的个体
            new_group = cur_group[:n_well]
            pbar.set_description((f'%-10s' + '%-10.4g') % (prefix, fitness[0]))
            # 使用轮盘赌法进行筛选
            proba = fitness - fitness.min()
            proba = proba / proba.sum()
            for pc, pv in np.random.random([n_choose, 2]):
                unit = random.choices(cur_group, weights=proba)[0].copy()
                # 交叉遗传 / 基因突变
                if pc <= self._cross_proba:
                    unit = self.cross(unit, random.choices(cur_group, weights=proba)[0].copy())
                if pv <= self._var_proba:
                    unit = self.variation(unit)
                # 拼接新个体
                new_group = np.concatenate([new_group, unit[None]])
            cur_group = new_group
        return cur_group[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    class ShortestPath(GeneticOpt):

        def init_adj(self):
            # 初始化邻接矩阵
            self.pos = np.random.random([self.n_gene, 2]) * 10
            self.adj = np.zeros([self.n_gene] * 2, dtype=DTYPE)
            for i in range(self.n_gene):
                for j in range(i + 1, self.n_gene):
                    self.adj[i][j] = self.adj[j][i] = \
                        np.sqrt(((self.pos[i] - self.pos[j]) ** 2).sum())

        def new_unit(self, size):
            ''' 初始化染色体群体'''
            group = []
            for _ in range(size):
                unit = list(range(self.n_gene))
                np.random.shuffle(unit)
                group += [unit]
            return np.array(group, dtype=np.int32)

        def fitness(self, group):
            ''' 适应度函数 (max -> best)'''
            group = np.concatenate([group, group[:, :1]], axis=-1)
            return - self.adj[group[:, :-1], group[:, 1:]].sum(axis=-1)


    ga = ShortestPath(80, 15, cross_proba=0, var_proba=0.6)
    ga.init_adj()
    unit = ga.fit(500)

    # 绘制最优路径
    fig = plt.subplot()
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    plt.plot(*ga.pos[unit].T, c='deepskyblue')
    plt.scatter(*ga.pos.T, marker='p', c='orange')
    plt.show()
