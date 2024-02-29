from typing import Sequence

import random
from optimize import *


def Eu_dist(data, center):
    """ 以 欧氏距离 为聚类准则的距离计算函数
        :param data: 形如 [n_sample, n_feature] 的 tensor
        :param center: 形如 [n_cluster, n_feature] 的 tensor"""
    return ((data[:, None] - center[None]) ** 2).sum(dim=2)


class Dist_Cluster:
    """ 基于距离的聚类器
        :param n_cluster: 簇中心数
        :param dist_fun: 距离计算函数
            :param data: 形如 [n_sample, n_feather] 的 tensor
            :param center: 形如 [n_cluster, n_feature] 的 tensor
            :return: 形如 [n_sample, n_cluster] 的 tensor
        :param mode: 距离优化模式 ("max", "mean", "sum")
        :param init: 初始簇中心
        :param patience: 允许 loss 无进展的次数
        :param lr: 中心点坐标学习率

        :ivar cluster_centers: 聚类中心
        :ivar labels: 聚类结果"""

    def __init__(self, n_cluster: int,
                 dist_fun: Callable[[torch.tensor, torch.tensor],
                                    torch.tensor] = Eu_dist,
                 mode: str = "max",
                 init: Optional[Sequence[Sequence]] = None,
                 patience: int = 50,
                 lr: float = 0.08):
        self._n_cluster = n_cluster
        self._dist_fun = dist_fun
        self._patience = patience
        self._lr = lr
        self._mode = mode
        # 初始化参数
        self.cluster_centers = None if init is None else torch.tensor(init).float()
        self.labels = None
        self._bar_len = 20

    def fit(self, data: torch.tensor, prefix="Cluster"):
        """ :param data: 形如 [n_sample, n_feature] 的 tensor
            :return: 簇惯性"""
        LOGGER.info(("%10s" * 3) % ("", "cur_loss", "min_loss"))
        self._init_cluster(data, self._patience // 5, prefix)
        inertia = self._train(data, self._lr, self._patience, prefix=prefix)
        # 开始若干轮次的训练，得到簇惯性
        self.classify(data)
        return inertia

    def classify(self, data: torch.tensor):
        """ :param data: 形如 [n_sample, n_feature] 的 tensor
            :return: 分类标签"""
        dist = self._dist_fun(data, self.cluster_centers)
        # 将标签加载到实例属性
        self.labels = dist.argmin(axis=1)
        return self.labels

    def _init_cluster(self, data, patience, prefix):
        # 没有中心点时，初始化一个中心点
        if self.cluster_centers is None:
            self.cluster_centers = data.mean(dim=0).reshape(1, -1)
        # 补全中心点
        for cur_center_num in range(self.cluster_centers.shape[0], self._n_cluster):
            dist = self._dist_fun(data, self.cluster_centers).min(dim=1)[0].cpu()
            dist -= dist.min()
            new_cluster = data[random.choices(range(data.shape[0]), weights=dist / dist.sum())].reshape(1, -1)
            # 取新的中心点
            self.cluster_centers = torch.cat([self.cluster_centers, new_cluster], dim=0).float()
            self._train(data, self._lr * 2.5, patience, prefix=f"Init_{cur_center_num}".ljust(len(prefix)), init=True)
            # 初始化簇中心时使用较大的lr

    def _train(self, data, lr, patience, prefix, init=False):
        loss_fun = lambda center: self._loss(data, center)
        self.cluster_centers, interia, _ = minimize(self.cluster_centers, loss_fun, lr=lr, patience=patience,
                                                    max_iter=None, prefix=prefix, title=False, leave=not init)
        return interia

    def _loss(self, data, center):
        sample_dist = self._dist_fun(data, center)
        min_dist, self.labels = sample_dist.min(dim=1)
        # 按照距离进行分类
        clf_result = []
        for idx in range(len(center)):
            own = self.labels == idx
            if torch.any(own):
                clf_result.append(min_dist[self.labels == idx])
            else:
                clf_result.append(sample_dist[:, idx].min())
        # 计算 loss 值
        if self._mode == "max":
            loss = sum([dists.max() + .05 * dists.mean() for dists in clf_result])
        elif self._mode == "mean":
            loss = sum([dists.mean() for dists in clf_result])
        elif self._mode == "sum":
            loss = sum([dists.sum() for dists in clf_result])
        else:
            raise KeyError("mode 参数出错")
        return loss


def Cos_similarity(data, refer):
    """ 余弦相似度计算
        :param data: 形如 [n_sample, n_feature] 的 tensor
        :param refer: 形如 [n_cluster, n_feature] 的 tensor"""
    data_len = (data ** 2).sum(dim=1) ** 0.5
    refer_len = (refer ** 2).sum(dim=1) ** 0.5
    # 计算向量模
    vec_dot = (data[:, None] * refer[None]).sum(dim=-1)
    # 计算余弦相似度
    return - vec_dot / (data_len[:, None] * refer_len[None])


def PIoU_dist(boxes, anchor, eps=1e-5):
    """ 以 IoU 为聚类准则的距离计算函数
        :param boxes: 形如 [n_sample, 2] 的 tensor
        :param anchor: 形如 [n_cluster, 2] 的 tensor"""
    boxes = boxes[:, None]
    anchor = anchor[None]
    max_coord = torch.maximum(boxes, anchor)
    min_coord = torch.minimum(boxes, anchor)
    # 计算交并面积
    inter = torch.prod(torch.relu(min_coord) + eps, dim=-1)
    union = torch.prod(max_coord, dim=-1)
    # 计算惩罚系数
    punish = torch.prod(torch.relu(- min_coord) + 1, dim=-1)
    # 计算 RIoU 损失
    return punish * (1 - inter / union)


def DIoU_dist(boxes, anchor):
    """ 以 DIoU 为聚类准则的距离计算函数
        :param boxes: 形如 [n_sample, 2] 的 tensor
        :param anchor: 形如 [n_cluster, 2] 的 tensor"""
    anchor = torch.abs(anchor)
    # 计算欧式距离
    dist = Eu_dist(boxes, anchor)
    boxes = boxes[:, None]
    anchor = anchor[None]
    # 计算交并面积
    union_and_inter = torch.prod(boxes, dim=-1) + torch.prod(anchor, dim=-1)
    inter = torch.prod(torch.minimum(boxes, anchor), dim=-1)
    iou = inter / (union_and_inter - inter)
    # 计算对角线长度
    diag = (torch.maximum(boxes, anchor) ** 2).sum(dim=-1)
    return 1 - iou + dist / diag


def cluster_plot_2d(cluster, data, opacity=0.5):
    centers = cluster.cluster_centers
    # 绘制样本点
    label = cluster.classify(data)
    for i in range(len(centers)):
        sample = data[label == i]
        plt.scatter(sample[:, 0], sample[:, 1], alpha=opacity)
    # 绘制聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], marker="p", color="gold")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.cluster import KMeans
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")


    class Timer:

        def __new__(cls, fun, *args, **kwargs):
            import time
            t0 = time.time()
            fun(*args, **kwargs)
            cost = time.time() - t0
            return cost * 1e3  # ms


    def draw_result(train_x, labels, cents, idx, title):
        """ 聚类结果可视化"""
        # 创建子图, 并设置标题
        fig = plt.subplot(1, 2, idx, projection="3d")
        plt.title(title)
        # 簇中心数量, 以及每个簇的颜色
        n_clusters = np.unique(labels).shape[0]
        color = ["red", "orange", "yellow"]
        # 分别绘制每个类别的样本
        for i in range(n_clusters):
            samples = train_x[labels == i]
            fig.scatter(*samples[:, :3].T, c=color[i])
        fig.scatter(*cents.T, c="deepskyblue", marker="*", s=100)
        # 视角变换
        fig.view_init(60, 110)


    # 读取数据集
    iris = datasets.load_iris()
    iris_x = torch.tensor(iris.data)
    iris_y = torch.tensor(iris.target, dtype=torch.int)

    # 使用 KMeans 聚类
    clf = KMeans(n_clusters=3)
    print(f"Kmeans: {Timer(clf.fit, iris_x):.0f} ms")
    draw_result(iris_x, clf.labels_, clf.cluster_centers_, 1, "KMeans++")

    # 使用基于距离的自定义聚类
    clf = Dist_Cluster(n_cluster=3, dist_fun=Eu_dist, lr=1., mode="mean")
    print(f"My Cluster: {Timer(clf.fit, iris_x):.0f} ms")
    draw_result(iris_x, clf.labels, clf.cluster_centers, 2, "My Cluster")

    plt.show()
