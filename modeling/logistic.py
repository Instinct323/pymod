import pickle

import numpy as np
from tqdm import tqdm

from utils import LOGGER


class FocalLoss:

    def __init__(self, gamma=1.5, alpha=0.5):
        self.gamma = gamma
        self.alpha = alpha
        self.save_for_backward = None

    def __call__(self, pred, target, eps=1e-4):
        # pred: [0, 1]
        pred = np.clip(pred, a_min=eps, a_max=1 - eps)
        log = target * np.log(pred) + (1 - target) * np.log(1 - pred)
        # FocalLoss 论文中提出的调制系数
        factor = target * (1 - pred) + (1 - target) * pred
        alpha = target * self.alpha + (1 - target) * (1 - self.alpha)
        self.save_for_backward = pred, target, log, factor, alpha
        return - log * factor ** self.gamma * alpha

    def backward(self, max_ce_grad=50):
        pred, target, log, factor, alpha = self.save_for_backward
        factor_clip = np.clip(factor, a_min=1 / max_ce_grad, a_max=1 - 1 / max_ce_grad)
        alpha *= np.sign(target - 0.5)
        diff = alpha * (self.gamma * factor ** (self.gamma - 1) * log -
                        factor ** self.gamma / (1 - factor_clip))
        self.save_for_backward = None
        return diff


def lazy_obj(file, fget, *args, **kwargs):
    if file.is_file():
        with open(file, 'rb') as pkf:
            data = pickle.load(pkf)
    else:
        data = fget(*args, **kwargs)
        with open(file, 'wb') as pkf:
            pickle.dump(data, pkf)
    return data


class Logistic_Regression:
    ''' 逻辑回归
        model_file: 参数文件
        lr: 学习率
        weight_decay: 正则化强度'''
    dtype = np.float16

    def __init__(self,
                 n_features,
                 model_file,
                 lr: float = 1.,
                 weight_decay: float = 2e-3,
                 gamma: float = 1.5,
                 alpha: float = 0.5):
        self.model_file = model_file
        self._weight_decay = weight_decay
        self._fl = FocalLoss(gamma=gamma, alpha=alpha)
        # 模型参数
        self._lr = lr
        self.weight, self.bias = lazy_obj(
            model_file, fget=lambda: [np.random.random(n_features).astype(self.dtype),
                                      np.array(0, dtype=self.dtype)]
        )

    def loss(self, data_set, train=True):
        features, target = map(lambda x: x.astype(self.dtype), data_set)
        score = features @ self.weight + self.bias
        pred = 1 / (1 + np.exp(-score))
        loss = self._fl(pred, target).mean()
        # 反向传播梯度, 更新变量
        if train:
            # d(l) / d(z)
            dl_dz = self._fl.backward() * np.exp(-score) * pred ** 2
            # 进行更新, 对 w 使用权重衰减
            self.weight -= self._lr * (features * dl_dz[:, None]).mean(axis=0)
            self.weight *= 1 - self._weight_decay
            self.bias -= self._lr * dl_dz.mean()
        return loss

    def fit(self, epochs, train_set, eval_set=None):
        ''' train_set: [train_x, train_y]
            eval_set: [eval_x, eval_y]'''
        # 记录当前最小 loss
        min_loss = self.loss(eval_set if eval_set else train_set, train=False)
        LOGGER.info(('%10s' + '%10s' * 3) % ('', 'Train', 'Eval', 'Min'))
        # 开始训练
        pbar = tqdm(range(epochs), total=epochs)
        for epoch in pbar:
            train_loss = self.loss(train_set, train=True)
            # 比较取最优
            eval_loss = self.loss(eval_set, train=False) if eval_set else train_loss
            if eval_loss < min_loss:
                min_loss = eval_loss
                with open(self.model_file, 'wb') as pkf:
                    pickle.dump([self.weight, self.bias], pkf)
            # 使用进度条输出损失信息
            pbar.set_description(('%10s' + '%10.6f' * 3) % ('Logistic', train_loss, eval_loss, min_loss))
        # 读取最优参数
        with open(self.model_file, 'rb') as pkf:
            self.weight, self.bias = pickle.load(pkf)
        return self

    def predict_proba(self, x):
        score = x @ self.weight + self.bias
        proba = 1 / (1 + np.exp(-score))
        return proba

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(np.int32)

    def score(self, x, y):
        return (self.predict(x) == y).sum() / len(y)

    def stability(self, x, pos=True):
        score = x @ self.weight + self.bias
        # 在只对一个变量进行放缩的情况下, 使得 P=0.5 的偏置
        bias = - score[:, None] / self.weight
        return np.abs(bias) if pos else bias


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from pathlib import Path

    # 模型的参数文件
    MODEL_FILE = Path('logistic.pkf')

    x, y = load_iris(return_X_y=True)
    # 将数据特征标准化
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    # 只选择类别 0, 1 的数据
    mask = y <= 1
    x, y = map(lambda z: z[mask], [x, y])

    # 初始化并开始训练
    clf = Logistic_Regression(
        n_features=x.shape[1], model_file=MODEL_FILE
    ).fit(epochs=100, train_set=[x, y])

    print(f'Weight: {clf.weight}')
    print(f'Accuracy: {clf.score(x, y)}')
    print(f'Proba: {clf.predict_proba(x)}')
