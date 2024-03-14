from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import make_grid


class FourierFeatures:
    """ :param f: Fourier basis frequencies"""

    def __init__(self,
                 c2: Optional[int] = None,
                 f: float = 20.,
                 seed: int = 0):
        assert c2 is None or not c2 & 1, "The number of output channels must be even"
        np.random.seed(seed)
        self.f = np.random.normal(0, f, (2, c2 // 2)) if c2 else None

    def __call__(self, w: int, h: int):
        x = 2 * np.pi * make_grid(w, h, percent=True, center=False)
        x = (x @ self.f) if isinstance(self.f, np.ndarray) else x
        return np.concatenate((np.sin(x), np.cos(x)), axis=-1)


def embedding_similarity(fe, w, h):
    x = fe(w, h)
    x = x.reshape(-1, x.shape[-1])
    # 计算余弦相似度
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    similarity = (x[:, None] * x).sum(axis=-1).reshape(-1, h, w)
    # 绘制余弦相似度
    fig, subs = plt.subplots(h, w)
    for i, sub in enumerate(subs.flatten()):
        for j in "xy": getattr(sub, f"set_{j}ticks")([])
        sub.imshow(similarity[i])
    plt.show()


def image_regression(image, fe, downsample=2., e=0.5,
                     epochs=1000, lr=5e-3, weight_decay=0., file=None):
    hi, wi, ci = image.shape
    h, w = np.round(np.array((hi, wi)) / downsample).astype(np.int64)
    from_np = lambda xi: torch.from_numpy(xi).cuda().float().permute(2, 0, 1)
    # 创建训练图像, 得到 fourier embedding
    image_train = from_np(cv2.resize(image, (w, h))) / 255
    x = from_np(fe(w, h))[None]
    # 创建模型和迭代器
    from .common import Conv
    c1, c2 = x.shape[1], round(x.shape[1] * e)
    model = nn.Sequential(
        Conv(c1, c2, 1, act=nn.ReLU),
        Conv(c2, ci, 1, act=nn.Sigmoid)
    ).cuda().float()
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.8, patience=20)
    # 使用 MSE 开始训练
    pbar = tqdm(range(epochs))
    for _ in pbar:
        y = model(x)[0]
        mse = F.mse_loss(y, image_train)
        mse.backward()
        pbar.set_description(f"MSE {mse * 255:.3f}")
        optim.step(), optim.zero_grad(), sche.step(mse)
    Conv.reparam(model)
    model.eval()
    # 绘制对比图像
    x = from_np(fe(wi, hi))[None]
    y = (model(x)[0] * 255).round().byte().cpu().permute(1, 2, 0).numpy()
    for i, img in enumerate((image, y)):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img[..., ::-1])
    # 保存图像 / 展示图像
    (plt.savefig(file), plt.close(), print(f"The result is stored in {file}")) if file else plt.show()
    return image, y


if __name__ == "__main__":
    np.set_printoptions(3, suppress=True)

    fe = FourierFeatures(128, f=20)
    # embedding_similarity(fe, 7, 7)

    img = cv2.imread(r"D:\Workbench\data\120a.jpg")
    image_regression(cv2.resize(img, (128,) * 2), fe, downsample=1, file="fourier.png")
