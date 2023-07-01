from typing import Union, List, Callable

import numpy as np


def softmax(x, axis=-1):
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


def rbf_affinity(x):
    k = x.shape[0] // 2
    dist = np.linalg.norm(x - x[:, None], axis=-1)
    sigma = np.sort(dist, axis=-1)[:, k].mean()
    return np.exp(- dist ** 2 / (2 * sigma ** 2))


class LAME:

    def __init__(self,
                 nc: Union[int, List[int]],
                 affinity: Callable = rbf_affinity,
                 max_iter: int = 100):
        self.nc = nc if isinstance(nc, int) else np.cumsum([1 if x == 2 else x for x in nc])
        self.affinity = affinity
        self.max_iter = max_iter

    def laplacianOptim(self, logp, kernel):
        entropy = float('inf')
        p = softmax(logp)
        for i in range(self.max_iter):
            # 根据亲和力对 p 进行加权, 作为 logp 的迭代值
            x = logp + kernel @ p
            p = softmax(x)
            # 计算 softmax(x) 与 log(x) 之间的 KL 散度
            e = (p * (np.log(p.clip(1e-20)) - x)).sum().item()
            if i > 1 and abs(e - entropy) <= 1e-8 * abs(entropy): break
            entropy = e
        return p

    def __call__(self, p, feats):
        # p: 概率矩阵 [B, nc]
        logp = np.log(p + 1e-10)
        kernel = self.affinity(feats / np.linalg.norm(feats, axis=-1, keepdims=True))
        if isinstance(self.nc, int):
            return self.laplacianOptim(logp, kernel)
        else:
            assert p.shape[-1] == self.nc[-1]
            # 类别数大于 2 时才会求解
            return np.concatenate([np.exp(x) if x.shape[-1] == 1 else self.laplacianOptim(x, kernel)
                                   for x in np.split(logp, self.nc, axis=-1)], axis=-1)


if __name__ == '__main__':
    np.set_printoptions(3, suppress=True)

    B = 4
    nc = 10

    p = np.random.random([B, nc])
    p[:, 2: 5] /= p[:, 2: 5].sum(axis=-1, keepdims=True)
    p[:, 5: 9] /= p[:, 5: 9].sum(axis=-1, keepdims=True)
    feats = np.random.normal(0, 1, [B, 10])

    print(p)
    lame = LAME([2, 2, 3, 4, 2])
    print(lame(p, feats))
