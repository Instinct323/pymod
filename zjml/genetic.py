import numpy as np
from tqdm import trange

DTYPE = np.float16


class Genetic_Algorithm:
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
        self._n_unit = n_unit
        self._n_gene = n_gene
        self._well_radio = well_radio
        self._cross_proba = cross_proba
        self._var_proba = var_proba
        self.group = self.new_unit(self._n_unit)

    def _random_section(self) -> tuple:
        ''' 产生随机区间'''
        gene_idx = list(range(self._n_gene))
        l = np.random.choice(gene_idx)
        r = np.random.choice(gene_idx[l:])
        return l, r

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
        l, r = self._random_section()
        np.random.shuffle(unit[l: r + 1])
        return unit

    def fitness(self, unit: np.ndarray) -> float:
        ''' 适应度函数 (max -> best)'''
        raise NotImplementedError

    def fit(self, epochs: int,
            patience: int = np.inf,
            prefix: str = 'GA_fit') -> np.ndarray:
        ''' :param epochs: 训练轮次
            :param patience: 允许搜索无进展的次数'''
        unit_idx = list(range(self._n_unit))
        pbar = trange(epochs)
        last_fitness, angry = - np.inf, 0
        # 最优个体数, 随机选取数
        n_well = round(self._n_unit * self._well_radio)
        n_choose = self._n_unit - n_well
        for _ in pbar:
            self.group = np.unique(self.group, axis=0)
            # 计算每个个体的适应度并排序
            fitness = np.array(list(map(self.fitness, self.group)), dtype=DTYPE)
            order = np.argsort(fitness)[::-1]
            # 收敛检测
            cur_fitness = fitness[order[0]]
            angry = 0 if cur_fitness > last_fitness else angry + 1
            last_fitness = cur_fitness
            if angry == patience: break
            # 保留一定数量的个体
            new_group = self.group[order[:n_well]]
            pbar.set_description((f'%-10s' + '%-10.4g') % (prefix, cur_fitness))
            fitness -= fitness.min()
            # 根据适应度, 使用轮盘赌法进行筛选
            proba = fitness / fitness.sum()
            choose_idx = np.random.choice(unit_idx[:len(self.group)], size=n_choose, p=proba)
            # 交叉遗传 / 基因突变
            for unit, (pc, pv) in zip(self.group[choose_idx], np.random.random([n_choose, 2])):
                if pc <= self._cross_proba:
                    unit = self.cross(unit, self.group[np.random.choice(unit_idx[:len(self.group)], p=proba)])
                if pv <= self._var_proba:
                    unit = self.variation(unit)
                # 拼接新个体
                new_group = np.concatenate([new_group, unit[None]])
            self.group = new_group
        return self.group[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    class Shortest_Path(Genetic_Algorithm):

        def new_unit(self, size):
            ''' 初始化染色体群体'''
            group = []
            for _ in range(size):
                unit = list(range(self._n_gene))
                np.random.shuffle(unit)
                group += [unit]
            return np.array(group, dtype=np.int32)

        def fitness(self, unit):
            ''' 适应度函数 (max -> best)'''
            # 初始化邻接表
            if not hasattr(self, 'adj'):
                self.pos = np.random.random([self._n_gene, 2]) * 10
                self.adj = np.zeros([self._n_gene] * 2, dtype=DTYPE)
                for i in range(self._n_gene):
                    for j in range(i + 1, self._n_gene):
                        self.adj[i][j] = self.adj[j][i] = \
                            np.sqrt(((self.pos[i] - self.pos[j]) ** 2).sum())
            # 计算适应度
            fitness = 0
            for i in range(self._n_gene - 1):
                dist = self.adj[unit[i]][unit[i + 1]]
                fitness += dist
            return - fitness


    np.random.seed(0)
    ga = Shortest_Path(80, 15, cross_proba=0, var_proba=0.6)
    unit = ga.fit(500)

    # 绘制最优路径
    fig = plt.subplot()
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    plt.plot(*ga.pos[unit].T, c='deepskyblue')
    plt.scatter(*ga.pos.T, marker='p', c='orange')
    plt.show()
