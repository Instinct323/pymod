import math
from typing import Optional, Callable

import numpy as np


class RANSAC(dict):
    """ Random Sample Consensus (RANSAC) algorithm
        :param n: number of minimum samples
        :param t: threshold value
        :param k: number of iterations
        :param w: probability of at least one sample is free from outliers
        :param p: probability of success

        Usage:

        for i in RANSAC.index_sampler(len(DATA)):
            MODEL.fit(DATA[i])
            error = MODEL.error(DATA)
            RANSAC.save_if_better(error, MODEL.export)"""

    def __init__(self,
                 n: int,
                 t: float,
                 k: Optional[int] = None,
                 w: Optional[float] = None,
                 p: Optional[float] = None):
        self.n = n
        self.t = t
        self.k = k
        if k is None:
            assert 0 < w <= 1, "w must be in (0, 1]"
            assert 0 < p <= 1, "p must be in (0, 1]"
            self.k = int(math.log(1 - p) / math.log(1 - w ** n))
        # best model
        super().__init__(params=None, inliers=None, fitness=0)

    def index_sampler(self, total: int):
        for _ in range(self.k):
            yield np.random.choice(total, min(total, self.n), replace=False)

    def save_if_better(self,
                       error: np.ndarray,
                       params_export: Callable):
        """ :param error: error of current model for all samples
            :param params_export: export current model parameters"""
        inliers = error < self.t
        fitness = inliers.mean()
        if fitness > self["fitness"]:
            self.update(params=params_export(), inliers=inliers, fitness=fitness)


if __name__ == '__main__':
    ransac = RANSAC(100, 5.991, 100)
    for i in ransac.index_sampler(1000):
        print(i)
