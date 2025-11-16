import torch
from torch import nn

from . import pointnet2_utils as pnu


class PointNet2(nn.ModuleList):
    ssg_config = dict(
        c2s_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        n2_list=[512, 128, 1],
        r_list=[0.2, 0.4, None],
        k_list=[32, 64, None]
    )

    msg_config = dict(
        c2s_list=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                  [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                  [256, 512, 1024]],
        n2_list=[512, 128, 1],
        r_list=[[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], None],
        k_list=[[16, 32, 128], [32, 64, 128], None]
    )

    def __init__(self,
                 c1: int,
                 c2s_list: list[list[int | list[int]]],
                 n2_list: list[int],
                 r_list: list[float | list[float]],
                 k_list: list[int | list[int]]):
        super().__init__()
        assert c1 >= 3, "Input channel must be at least 3 (xyz)."

        self.c2s = [c1 - 3]
        for args in zip(n2_list, r_list, k_list, c2s_list):
            c2 = args[-1]
            if isinstance(c2[0], list):
                self.append(pnu.PointNetSetAbstractionMsg(self.c2s[-1], *args))
                self.c2s.append(sum(c[-1] for c in c2))
            else:
                self.append(pnu.PointNetSetAbstraction(3 + self.c2s[-1], *args))
                self.c2s.append(c2[-1])

    def forward(self, xyz_feat) -> pnu.Latent:
        # xyz_feat: [B, N, C]
        B, _, N = xyz_feat.shape
        latent = pnu.Latent.from_tensor(xyz_feat)
        for sa in self: latent = sa(latent)
        return latent


if __name__ == '__main__':
    c1 = 3
    x = torch.rand([2, 1024, c1])

    model = PointNet2(c1, **PointNet2.msg_config)

    out = model(x)
    idx = out.latent_prev.idx_source
    mse = torch.square(out.latent_prev.xyz - pnu.index_points(x[..., :3], idx)).sum()
    print(mse)
