from dataclasses import dataclass, field

import torch
from torch import nn


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

        from . import pointnet2_utils as pnu
        self.utils = pnu

        self.c2s = [c1 - 3]
        for args in zip(n2_list, r_list, k_list, c2s_list):
            c2 = args[-1]
            if isinstance(c2[0], list):
                self.append(pnu.PointNetSetAbstractionMsg(self.c2s[-1], *args))
                self.c2s.append(sum(c[-1] for c in c2))
            else:
                self.append(pnu.PointNetSetAbstraction(3 + self.c2s[-1], *args))
                self.c2s.append(c2[-1])

    def forward(self, xyz_feat) -> "Latent":
        """
        :param xyz_feat: [B, C1, N]
        :return: (xyz[B, 3, N'], feat[B, C2, N'], fps[B, N']) list
        """
        B, _, N = xyz_feat.shape
        latent = self.Latent.from_tensor(xyz_feat)
        for sa in self:
            latent = self.Latent(*sa(*latent.as_input()), latent_prev=latent)
        return latent

    @dataclass
    class Latent:
        xyz: torch.Tensor  # [B, 3, N]
        feature: torch.Tensor = None  # [B, C, N]
        idx_prev: torch.Tensor = None  # [B, N]

        latent_prev: 'PointNet2.Latent' = field(default=None, repr=False)

        def __post_init__(self):
            if self.idx_prev is None:
                B, _, N = self.xyz.shape
                self.idx_prev = torch.arange(N, device=self.xyz.device)[None].repeat(B, 1)

        def as_input(self, concat: bool = False):
            args = self.xyz, self.feature
            return torch.concat(args, dim=1) if concat else args

        @classmethod
        def from_tensor(cls,
                        xyz_feat: torch.Tensor):
            args = (xyz_feat,) if xyz_feat.shape[1] == 3 else (xyz_feat[:, :3, :], xyz_feat[:, 3:, :])
            return cls(*args)

        @property
        def idx_source(self) -> torch.Tensor:
            idx = self.idx_prev
            if self.latent_prev:
                idx = torch.gather(self.latent_prev.idx_source, dim=1, index=idx)
            return idx


if __name__ == '__main__':
    c1 = 3
    x = torch.rand([2, c1, 1024])

    model = PointNet2(c1, **PointNet2.msg_config)
    model.pop(-1)

    out = model(x)
    idx = out.idx_source
    mse = torch.square(out.xyz.permute(0, 2, 1) -
                       model.utils.index_points(x[:, :3].permute(0, 2, 1), idx)).sum()
    print(mse)
