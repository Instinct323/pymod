import numpy as np
import torch


class DecisionWeight:
    """ 决策权重生成器
        :param xtype: 输入类型 (pcl_xyz, img_rgb) """

    def __init__(self, xtype: str = None):
        self.xtype: str = xtype
        self.x: torch.Tensor = None

    def set_input(self, x: torch.Tensor):
        self.x = x
        x.requires_grad_(True)

    def process(self, loss: torch.Tensor) -> np.ndarray:
        assert isinstance(self.x, torch.Tensor), "Input tensor is not set."
        loss.backward()
        ret, self.x = self.x.grad.cpu().numpy(), None
        # 如果设置了 x 的类型, 则对应的处理
        if self.xtype in ("pcl_xyz",):
            # 点云的 xyz 位移
            ret = np.linalg.norm(ret, axis=-1)
        elif self.xtype in ("img_rgb",):
            # 图像的像素灰度变化
            ret *= np.array([0.299, 0.587, 0.114])[..., None, None]
            ret = np.abs(ret.sum(axis=-3))
        else:
            assert ret is None, f"Unsupported xtype: {self.xtype}"
        return ret

    @staticmethod
    def to_rgb(w: np.ndarray,
               nstd: float = 2.,
               thresh: float = 0.) -> np.ndarray:
        """ 将权重映射为 RGB
            :param w: 权重 [0, inf)
            :param nstd: 标准差倍数
            :param thresh: 标准差倍数阈值 """
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("rainbow")
        # 除以 n 倍标准差
        w /= np.sqrt(np.square(w).mean())
        w[w < thresh] = 0
        w = np.clip(w / nstd, 0, 1)
        return np.uint8(255 * cmap(w)[..., :3])
