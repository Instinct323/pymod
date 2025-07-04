from typing import Type

import numpy as np
from tqdm import trange


class ChromosomeBase:
    """ 染色体基类"""

    def __eq__(self, other):
        """ 重载 == 运算符, 用于去重"""
        raise NotImplementedError

    def fitness(self) -> float:
        """ 适应度函数 (max -> best)"""
        raise NotImplementedError

    def mutation(self):
        """ 基因突变"""
        raise NotImplementedError

    def crossover(self, other):
        """ 交叉遗传"""
        raise NotImplementedError


class GeneticOpt:
    """
    遗传算法
    :param cross_proba: 交叉概率
    :param mut_proba: 变异概率
    :param well_radio: 最优个体比例
    :ivar group: 染色体群体
    """

    def __init__(self,
                 chromosome: Type[ChromosomeBase],
                 n_unit: int,
                 cross_proba: float = 0.4,
                 mut_proba: float = 0.3,
                 well_radio: float = 0.2,
                 best_unit: ChromosomeBase = None):
        self.chromosome = chromosome
        self.n_unit = n_unit
        self.group = self.new_unit(self.n_unit)
        self.log = []

        assert 0 < cross_proba + mut_proba < 1, "cross_proba + mut_proba must be in (0, 1)"
        self._cross_proba = cross_proba
        self._mut_proba = mut_proba

        assert 0 < well_radio < 1, "well_radio must be in (0, 1)"
        self._well_radio = well_radio

        if isinstance(best_unit, ChromosomeBase): self.group[0] = best_unit

    def new_unit(self, size: int) -> list:
        """ 初始化染色体群体"""
        return [self.chromosome() for _ in range(size)]

    def sort_unique(self, group: list):
        """ 计算每个染色体的适应度, 排序、去重"""
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
            prefix: str = "GA-fit"):
        """
        :param epochs: 训练轮次
        :param patience: 允许搜索无进展的次数
        :param prefix: 进度条前缀
        """
        pbar = trange(epochs)
        angry = 0
        # 最优个体数, 随机选取数
        n_well = max(2, round(self.n_unit * self._well_radio))
        n_choose = self.n_unit - n_well
        # 轮盘赌法采样
        f_sample = lambda p: np.random.choice(self.group, p=p)
        for _ in pbar:
            self.group, fitness = self.sort_unique(self.group)
            # 收敛检测
            if self.log and np.isfinite(self.log[-1][0]):
                angry = 0 if fitness[0] > self.log[-1][0] else angry + 1
                if angry == patience: break
            self.log.append([fitness[0], fitness.mean(), fitness.std(), len(fitness)])
            # 保留一定数量的个体
            tmp_group = self.group[:n_well]
            pbar.set_description((f"%-10s" + "%-10.4g") % (prefix, fitness[0]))
            # 使用轮盘赌法进行筛选
            p = fitness - fitness.min()
            p = p / p.sum()
            for x in np.random.random(n_choose):
                unit = f_sample(p)
                # 基因突变 / 交叉遗传
                if x <= self._mut_proba:
                    unit = unit.mutation()
                elif 1 - x <= self._cross_proba:
                    unit = unit.crossover(f_sample(p))
                tmp_group.append(unit)
            self.group = tmp_group
        pbar.close()
        return self.group[0], pd.DataFrame(self.log, columns=["fit-best", "fit-mean", "fit-std", "n-unique"])


class TspPath(ChromosomeBase):
    """ 旅行商问题"""
    n = property(lambda self: len(self.adj))
    adj = None
    cluster = None

    @classmethod
    def kmeans_init(cls):
        k = round(np.sqrt(len(cls.adj)))
        # 对邻接矩阵进行聚类
        from sklearn.cluster import KMeans
        clf = KMeans(n_clusters=k, n_init="auto")
        clf.fit(cls.adj)
        # 路标点分块
        cls.cluster = [np.where(clf.labels_ == i)[0] for i in range(k)]
        print("Init cluster.")

    def __init__(self, data=None):
        # self.data = data if isinstance(data, np.ndarray) else np.random.permutation(self.n)
        if data is not None:
            self.data = data
        else:
            if self.cluster:
                # 分块随机的路径序列
                np.random.shuffle(self.cluster)
                tuple(map(np.random.shuffle, self.cluster))
                self.data = np.concatenate(self.cluster)
            else:
                # 完全随机的路径序列
                self.data = np.random.permutation(self.n)

        data = np.concatenate([self.data, self.data[:1]])
        self._dist = self.adj[data[:-1], data[1:]]
        self._p = self._dist / self._dist.sum()

    def __eq__(self, other):
        return np.all(self.data == other.data)

    def fitness(self) -> float:
        return - self._dist.sum()

    def mutation(self):
        """ 基因突变"""
        l = np.random.randint(0, self.n - 1)
        r = np.random.randint(l + 1, self.n)
        # note: 对数据进行深拷贝, 否则会影响其他个体
        data = self.data.copy()

        tmp = self.adj[data[l - 1], data[r]] + self.adj[data[l], data[(r + 1) % self.n]]
        cur = self.adj[data[l - 1], data[l]] + self.adj[data[r], data[(r + 1) % self.n]]

        if tmp < cur:
            data[l: r + 1] = data[l: r + 1][::-1]
        else:
            np.random.shuffle(data[l: r + 1])
        return __class__(data)

    def crossover(self, other):
        """ 交叉遗传"""
        other = other.data.tolist()

        for _ in range(10):
            l = np.random.choice(self.n, p=self._p)

            lo = other.index(self.data[l])
            ro = other.index(self.data[(l + 1) % self.n])

            if not (2 < abs(lo - ro) < self.n / 2): continue

            other = other[lo: ro + 1] if lo < ro else other[ro: lo + 1][::-1]
            return __class__(np.array(
                [x for x in self.data[:l] if x not in other] +
                other + [x for x in self.data[l + 2:] if x not in other]
            ))
        return self.mutation()


if __name__ == "__main__":
    from pymod.zjplot import *

    np.random.seed(0)

    N_NODE = 150
    # 初始化邻接矩阵
    POS = np.random.random([N_NODE, 2]) * 10
    ADJ = TspPath.adj = np.sqrt(np.square(POS[:, None] - POS).sum(axis=-1))

    colors = [blue, purple]
    labels = ["Random", "Proposed"]

    for i in range(2):
        if i: TspPath.kmeans_init()

        ga = GeneticOpt(TspPath, 50, cross_proba=0.4 * i, mut_proba=0.3 * (2 - i))
        unit, log = ga.fit(2500)
        unit = np.concatenate([unit.data, unit.data[:1]])

        # 绘制最优路径
        fig = plt.subplot(1, 3, i + 1)
        plt.title(labels[i])
        plt.xticks([]), plt.yticks([])
        for key in "right", "top":
            fig.spines[key].set_color("None")
        plt.plot(*POS[unit].T, c=colors[i])
        plt.scatter(*POS.T, marker="p", c="orange")

        # 绘制适应度曲线
        plt.subplot(1, 3, 3)
        regionplot(log["fit-best"], log["fit-mean"], log["fit-std"], y_color=colors[i], label=labels[i])

    plt.subplot(1, 3, 3)
    plt.title("fitness")
    plt.legend()
    plt.show()
