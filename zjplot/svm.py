from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def cal_dist(x, coef: np.array, bias: Union[int, float]):
    ''' 计算点到直线的距离 (保留正负号)
        Dist = (coef × x + bias) / ||coef||
        :param x: [n_sample, n_dim]
        :param coef: [n_dim, ]'''
    return (x @ coef + bias) / np.linalg.norm(coef, ord=2)


def plot_hyperplane(svc, dataset, label, fig=None, s=None,
                    scatter_color=['deepskyblue', 'orange'],
                    plane_color=['mediumpurple', 'violet']):
    ''' 二分类 SVM 可视化
        :param svc: 线性支持向量机实例
        :param dataset: 数据集, [n_sample, n_dim]
        :param label: 数据标签, [n_sample, ]
        :param scatter_color: 负样本、正样本散点颜色
        :param plane_color: 分界超平面、极端超平面颜色'''
    # 读取超平面参数
    coef, bias = svc.coef_[0], svc.intercept_[0]
    # 各个维度的上下限
    n_dim = len(coef)
    limit = np.array([(dataset[:, i].min(), dataset[:, i].max())
                      for i in range(n_dim)])
    # 上下限扩充: 防止位于边界上的样本点被截掉
    extension = (limit[:, 1] - limit[:, 0]) * 0.1
    limit[:, 0] -= extension
    limit[:, 1] += extension
    # 绘制 2D 图像
    if n_dim == 2:
        fig = fig if fig else plt.subplot()
        coef_x, coef_y = coef
        # 找到直线的两个顶点
        x = limit[0]
        cal_y = lambda x, b: - (coef_x * x + b) / coef_y
        # 绘制分界直线
        if plane_color[0]:
            y = cal_y(x, bias)
            plt.plot(x, y, color=plane_color[0])
        # 绘制极端直线
        if plane_color[1]:
            for b_ in [-1, 1]:
                y = cal_y(x, bias + b_)
                plt.plot(x, y, color=plane_color[1], linestyle='--')
        # 裁剪画布边界
        for lim, func in zip(limit, [plt.xlim, plt.ylim]):
            func(lim)
    # 绘制 3D 图像
    elif n_dim == 3:
        fig, opacity = fig = fig if fig else plt.subplot(projection='3d'), 0.5
        coef_x, coef_y, coef_z = coef
        # 定义计算 z 的函数
        cal_z = lambda x, y, b: - (coef_x * x + coef_y * y + b) / coef_z

        def get_vex(b):
            x, y = np.meshgrid(*limit[:2])
            x, y = x.reshape(-1), y.reshape(-1)
            z = cal_z(x, y, b)
            # Δz: z - Δz ∈ [z_min, z_max]
            z_min, z_max = limit[2]
            delta_z = (z > z_max) * (z - z_max) + (z < z_min) * (z - z_min)
            # subject to: coef_x·Δx + coef_y·Δy + coef_z·Δz = 0
            delta_x = - coef_z * delta_z / coef_x
            x_ = x - delta_x
            delta_y = - coef_z * delta_z / coef_y
            y_ = y - delta_y
            # 获得新的点集
            x = np.stack([x, x_], axis=-1).reshape(-1)
            y = np.stack([y_, y], axis=-1).reshape(-1)
            # 剔除相同的点
            points = np.unique(np.stack([x, y], axis=-1), axis=0)
            points = np.concatenate([points, points[-1].reshape(1, -1)]) if len(points) & 1 else points
            # 定义平面的顶点
            x, y = points.T.reshape(2, 2, -1)[..., ::-1]
            return x, y, cal_z(x, y, b)

        # 绘制分界平面
        if plane_color[0]:
            fig.plot_surface(*get_vex(bias), alpha=opacity, color=plane_color[0])
        # 绘制极端平面
        if plane_color[1]:
            for b_ in [-1, 1]:
                fig.plot_surface(*get_vex(bias + b_), alpha=opacity, color=plane_color[1])
        # 裁剪画布边界
        for lim, func in zip(limit, [fig.set_xlim3d, fig.set_ylim3d, fig.set_zlim3d]):
            func(lim)
    else:
        raise AssertionError(f'不支持{n_dim}维数据的可视化')
    # 绘制样本散点
    if scatter_color:
        scatter_color = scatter_color * 2 if len(scatter_color) == 1 else scatter_color
        fig.scatter(*[dataset[:, i] for i in range(n_dim)],
                    color=[scatter_color[l] for l in map(int, label)], s=s)


if __name__ == '__main__':
    import random

    from sklearn.svm import SVC

    # 生成数据集
    train_set = np.stack([np.random.rand(1000) * 5
                          for _ in range(3)]).T
    cal = lambda x: cal_dist(x, coef=np.array([-0.5, -1, 2]), bias=-2)
    # 保留距离预定超平面 > 0.5 的点
    train_set = train_set[np.abs(cal(train_set)) > 0.5]
    # 根据距离的正负给定分类
    train_set_label = cal(train_set) > 0
    random.shuffle(train_set[:100])

    clf = SVC(kernel='linear')
    clf.fit(train_set, train_set_label)

    plot_hyperplane(clf, train_set, train_set_label)
    plt.show()
