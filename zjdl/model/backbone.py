from typing import List, Union

import torch
from torch import nn

from . import pointnet2_utils as pnu


class PointNet2(nn.ModuleList):
    SSG_CLS = dict(
        c2s_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        n2_list=[512, 128, 1],
        k_list=[32, 64, 0],
        r_list=[.2, .4, 0],
    )

    MSG_CLS = dict(
        c2s_list=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                  [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                  [256, 512, 1024]],
        n2_list=[512, 128, 1],
        k_list=[[16, 32, 128], [32, 64, 128], 0],
        r_list=[[.1, .2, .4], [.2, .4, .8], 0],
    )

    SSG_SEM = dict(
        c2s_list=[[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]],
        n2_list=[1024, 256, 64, 16],
        k_list=[32] * 4,
        r_list=[.1, .2, .4, .8],
        c2s_list_neck=[[256, 256], [256, 256], [256, 128], [128, 128]]
    )

    MSG_SEM = dict(
        c2s_list=[[[16, 16, 32], [32, 32, 64]],
                  [[64, 64, 128], [64, 96, 128]],
                  [[128, 192, 256], [128, 192, 256]],
                  [[256, 256, 512], [256, 384, 512]]],
        n2_list=[1024, 256, 64, 16],
        k_list=[[16, 32]] * 4,
        r_list=[[.05, .1], [.1, .2], [.2, .4], [.4, .8]],
        c2s_list_neck=[[512, 512], [512, 512], [256, 256], [128, 128]]
    )

    def __init__(self,
                 c1: int,
                 c2s_list: List[List[Union[int, List[int]]]],
                 n2_list: List[int],
                 k_list: List[Union[int, List[int]]],
                 r_list: List[Union[float, List[float]]],
                 drop_list: List[float] = None,
                 c2s_list_neck: List[List[int]] = None,
                 drop_list_neck: List[int] = None):
        super().__init__()
        assert c1 >= 3, "Input channel must be at least 3 (xyz)."
        self.c2s = [c1 - 3]

        if drop_list is None: drop_list = [0.] * len(c2s_list)
        for args in zip(c2s_list, n2_list, k_list, r_list, drop_list):
            c2 = args[0]
            SAtype = pnu.PointNetSetAbstractionMsg if isinstance(c2[0], list) else pnu.PointNetSetAbstraction
            self.append(SAtype(self.c2s[-1], *args))
            self.c2s.append(self[-1].c2)

        if c2s_list_neck is not None:
            if drop_list_neck is None: drop_list_neck = [0.] * len(c2s_list_neck)
            n_sa = len(self)
            for i in range(len(c2s_list_neck)):
                self.append(
                    pnu.PointNetFeaturePropagation(c11=self.c2s[-1], c12=self.c2s[n_sa - i - 1],
                                                   c2s=c2s_list_neck[i], drop=drop_list_neck[i])
                )
                self.c2s.append(self[-1].c2)

    def forward(self, xyz_feat) -> pnu.Latent:
        # xyz_feat: [B, N, C]
        latent = pnu.Latent.from_tensor(xyz_feat) if isinstance(xyz_feat, torch.Tensor) else xyz_feat
        for sa in self: latent = sa(latent)
        return latent


if __name__ == '__main__':
    c1 = 3
    x = torch.rand([2, 1024, c1])

    model = PointNet2(c1, **PointNet2.MSG_SEM)
    print(model)

    latent = model(x)
    idx = latent.latent_prev.idx_source
    print(latent.feature.shape)
    print(nn.functional.mse_loss(latent.latent_prev.xyz, pnu.index_points(x[..., :3], idx)))
