import numpy as np


class Crosstab:
    _div = lambda self, a, b, decimal=4, eps=1e-8: np.round(a / (b + eps), decimal)

    def __init__(self, pred, target, num_classes: int = -1):
        num_classes = max(target) + 1 if num_classes == -1 else num_classes
        empty = sum(len(y) for y in (pred, target)) == 0
        assert empty or all("int" in str(y.dtype) for y in (pred, target)), \
            "Only integer can be used to represent categories"
        if empty:
            self.data = np.zeros((num_classes,) * 2, dtype=np.int32)
        else:
            pred, target = map(lambda x: x.flatten(), (pred, target))
            self.data = np.bincount(pred + num_classes * target, minlength=num_classes ** 2).reshape((num_classes,) * 2)

    _tp = property(lambda self: np.diag(self.data))
    accuracy = property(lambda self: self._div(self._tp.sum(), self.data.sum()))
    precision = property(lambda self: self._div(self._tp, self.data.sum(axis=0)))
    recall = property(lambda self: self._div(self._tp, self.data.sum(axis=1)))

    def f_score(self, beta=1.):
        alpha = beta ** 2
        precision, recall = self.precision, self.recall
        return self._div((1 + alpha) * precision * recall, alpha * precision + recall)

    @property
    def kappa(self):
        x = self.data / self.data.sum()
        pe = (x.sum(0) * x.sum(1)).sum()
        return max(0, (self.accuracy - pe) / (1 - pe))

    def eval(self, beta=1.):
        return {"Accuracy": self.accuracy, "Precision": self.precision,
                "Recall": self.recall, "Kappa": self.kappa, f"F{beta:.1f}-Score": self.f_score(beta)}

    def __add__(self, other):
        if isinstance(other, __class__):
            self.data += other.data
            return self
        types = tuple(map(lambda x: type(x).__name__, (self, other)))
        raise TypeError(f"unsupported operand type(s) for +: \"{types[0]}\" and \"{types[1]}\"")

    def __repr__(self):
        return str(self.data)


if __name__ == "__main__":
    num_classes = 3

    pred = np.random.randint(0, num_classes, [100])
    target = np.random.randint(0, num_classes, [100])
    counter = Crosstab(pred, target)
    print(counter)
    # 累加一个新的混淆矩阵
    counter += Crosstab([], [], num_classes=num_classes)
    # 不同类型不能相加, 直接报错
    # counter += 1

    print(counter.kappa)
