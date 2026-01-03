# FROM: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
import copy
import re

import torch
import torch.nn as nn
from dataclasses import dataclass, field

from . import common


def align_pointnet2_state_dict(state_dict: dict) -> dict:
    """
    TODO: PointNetFeaturePropagation
    :param state_dict: https://github.com/erikwijmans/Pointnet2_PyTorch style state dict
    """
    ret = copy.copy(state_dict)
    for k in state_dict:
        rfind = re.search(r"(mlps\.[0-9]+)\.([0-9]+)\.(\w+$)", k)

        if rfind:
            stem = k.replace(rfind.group(), "").rstrip(".")
            mid = int(rfind.group(2))
            mname = f"{mid // 3}." + ["linear", "bn"][mid % 3]

            knew = ".".join([stem, rfind.group(1), mname, rfind.group(3)])
            ret[knew] = ret.pop(k)
    return ret


def square_distance(src: torch.Tensor,
                    dst: torch.Tensor) -> torch.Tensor:
    """
    :param src: source points, [B, N, C]
    :param dst: target points, [B, M, C]
    :return: squared distance, [B, N, M]
    """
    dist = -2 * src @ dst.permute(0, 2, 1)
    dist += (src ** 2).sum(dim=-1, keepdim=True)
    dist += (dst ** 2).sum(dim=-1)[:, None]
    return dist


def index_points(points: torch.Tensor,
                 idx: torch.Tensor) -> torch.Tensor:
    """
    :param points: input points data, [B, N, C]
    :param idx: sample index data, [B, ...]
    :return: indexed points data, [B, ..., C]
    """
    B, *idx_shape = idx.shape
    bi = torch.arange(B, dtype=torch.long).to(points.device).view([B] + [1] * len(idx_shape))
    bi, idx = torch.broadcast_tensors(bi, idx)
    return points[bi, idx, :]


def farthest_point_sample(xyz: torch.Tensor,
                          n2: int) -> torch.Tensor:
    """
    :param xyz: point cloud data, [B, N, 3]
    :param n2: number of samples
    :return: sampled point index, [B, n2]
    """
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, n2, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    bi = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n2):
        centroids[:, i] = farthest
        centroid = xyz[bi, farthest][:, None]
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1)[1]
    return centroids


def query_ball_point(xyz1: torch.Tensor,
                     xyz2: torch.Tensor,
                     k: int,
                     r: float) -> torch.Tensor:
    """
    :param xyz1: all points, [B, N, 3]
    :param xyz2: query points, [B, S, 3]
    :param k: max sample number in local region
    :param r: local region radius
    :return: grouped points index, [B, S, k]
    """
    B, N, _ = xyz1.shape
    _, S, _ = xyz2.shape

    # [B, S, N]
    mask_invalid = square_distance(xyz2, xyz1) > r ** 2
    new_idx = torch.arange(N, N + S).to(xyz1.device).view(1, -1, 1).repeat([B, 1, N])
    group_idx = torch.arange(N).to(new_idx).view(1, 1, -1).repeat([B, S, 1])
    group_idx[mask_invalid] = new_idx[mask_invalid]

    # [B, S, k]
    group_idx = torch.topk(group_idx, k=k, dim=-1, largest=False).values
    group_first = group_idx[:, :, :1].repeat([1, 1, k])
    mask = group_idx >= N
    group_idx[mask] = group_first[mask]
    return group_idx


def query_knn_point(xyz1: torch.Tensor,
                    xyz2: torch.Tensor,
                    k: int) -> torch.Tensor:
    """
    :param xyz1: all points, [B, N, 3]
    :param xyz2: query points, [B, S, 3]
    :param k: max sample number in local region
    :return: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(xyz2, xyz1)
    group_idx = torch.topk(sqrdists, k=k, dim=-1, largest=False, sorted=False).indices
    return group_idx


@dataclass
class Latent:
    xyz: torch.Tensor  # [B, N, 3]
    feature: torch.Tensor = None  # [B, N, C]
    idx_prev: torch.Tensor = None  # [B, N]

    latent_prev: 'Latent' = field(default=None, repr=False)

    def __getitem__(self, item):
        return Latent(
            xyz=index_points(self.xyz, item),
            feature=index_points(self.feature, item) if self.feature is not None else None,
            idx_prev=item,
            latent_prev=self
        )

    def as_input(self):
        args = self.xyz, self.feature
        return args[0] if args[1] is None else torch.cat(args, dim=-1)

    @classmethod
    def from_tensor(cls,
                    xyz_feat: torch.Tensor):
        args = (xyz_feat,) if xyz_feat.shape[-1] == 3 else (xyz_feat[..., :3], xyz_feat[..., 3:])
        return cls(*args)

    @property
    def idx_source(self) -> torch.Tensor:
        idx = self.idx_prev
        if idx is None:
            B, N, C = self.xyz.shape
            idx = torch.arange(N, device=self.xyz.device)[None].repeat(B, 1)

        if self.latent_prev:
            idx = torch.gather(self.latent_prev.idx_source, dim=1, index=idx)
        return idx


def sample_ops(src: Latent, n2: int):
    if n2 == 1:
        return Latent(xyz=torch.zeros_like(src.xyz[:, :1]).to(src.xyz), latent_prev=src)
    elif n2 > src.xyz.shape[1]:
        return Latent(xyz=src.xyz, latent_prev=src)
    else:
        fps = farthest_point_sample(src.xyz, n2=n2)
        return Latent(xyz=index_points(src.xyz, fps), idx_prev=fps, latent_prev=src)


def group_ops(src: Latent, dst: Latent, mlp: nn.Module, k: int, group_type: str, **kwargs):
    assert dst.feature is None
    # group all
    if k == 0:
        dst.feature = src.as_input()[:, None]
    else:
        group_type = {"ball": query_ball_point, "knn": query_knn_point}.get(group_type, group_type)
        group_latent = src[group_type(src.xyz, dst.xyz, k=k, **kwargs)]
        group_latent.xyz -= dst.xyz.unsqueeze(2)
        dst.feature = group_latent.as_input()
    dst.feature = mlp(dst.feature.permute(0, 3, 1, 2)).max(dim=-1)[0].permute(0, 2, 1)
    return dst


class ColorPointCloudAugmentation(nn.Module):

    def __init__(self,
                 hyp: dict = {
                     "xyz_jitter": {"std": 0.005, "clip": 0.02},
                     "color_jitter": {"brightness": .2, "contrast": .2, "saturation": .2, "hue": .05},
                     "color_drop": .2,
                 }):
        super().__init__()
        self.hyp = hyp

        if "color_jitter" in hyp:
            import torchvision.transforms as vtf
            hyp["color_jitter"] = vtf.ColorJitter(**hyp["color_jitter"])

    def xyz_jitter(self, latent: Latent):
        hyp = self.hyp["xyz_jitter"]
        noise = torch.randn_like(latent.xyz) * hyp["std"]
        noise = noise.clamp(-hyp["clip"], hyp["clip"])
        latent.xyz += noise
        return latent

    def color_drop(self, latent: Latent):
        latent.feature = nn.functional.dropout(latent.feature, p=self.hyp["color_drop"])
        return latent

    def color_jitter(self, latent: Latent):
        latent.feature = self.hyp["color_jitter"](
            latent.feature.permute(2, 0, 1)[None]
        )[0].permute(1, 2, 0)
        return latent

    def forward(self, latent: Latent):
        if self.training:
            latent = copy.copy(latent)
            for k in self.hyp:
                latent = getattr(self, k)(latent)
        return latent


class PointNetSetAbstractionMsg(nn.Module):

    def __init__(self, c1, c2s_list, n2, k_list, r_list, drop=0.):
        super().__init__()
        self.c2 = sum(c[-1] for c in c2s_list)
        self.n2 = n2
        self.k_list = k_list
        self.r_list = r_list
        self.mlps = nn.ModuleList([
            common.ConvBnAct2d.create_mlp(c1 + 3, c2s=c2s_list[i], k=1) for i in range(len(c2s_list))
        ])
        self.drop = nn.Dropout(drop)

    def extra_repr(self) -> str:
        attr = {"n2": self.n2, "k": self.k_list, "r": self.r_list}
        return ", ".join(f"{k}={v}" for k, v in attr.items())

    def forward(self, latent: Latent):
        ret = sample_ops(latent, n2=self.n2)
        feature_list = []
        for i, (k, r, mlp) in enumerate(zip(self.k_list, self.r_list, self.mlps)):
            feature_list.append(group_ops(latent, ret, mlp=mlp, k=k, r=r, group_type="ball").feature)
            ret.feature = None

        ret.feature = torch.cat(feature_list, dim=-1)
        ret.feature = self.drop(ret.feature)
        return ret


class PointNetSetAbstraction(PointNetSetAbstractionMsg):

    def __init__(self, c1, c2s, n2, k, r, drop=0.):
        super().__init__(c1, [c2s], n2, [k], [r], drop)
        self.group_all = n2 == 1
        assert not (self.group_all and any((k, r)))

    def extra_repr(self) -> str:
        attr = {"n2": self.n2}
        if not self.group_all: attr.update({"k": self.k, "r": self.r})
        return ", ".join(f"{k}={v}" for k, v in attr.items())


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, c11, c12, c2s, k=3, drop=0.):
        super().__init__()
        self.c2 = c2s[-1]
        self.k = k
        self.mlp = common.ConvBnAct1d.create_mlp(c11 + c12, c2s=c2s, k=1)
        self.drop = nn.Dropout(drop)

    def forward(self,
                src: Latent,
                dst: Latent = None) -> Latent:
        """
        :param src: queried points data
        :param dst: target points data
        :return: propagated points data
        """
        if dst is None: dst = src.latent_prev
        ret = copy.copy(dst)

        if src.xyz.shape[1] == 1:
            ret.feature = src.feature.repeat(1, dst.xyz.shape[1], 1)
        else:
            dists = square_distance(dst.xyz, src.xyz)
            dists, idx = dists.topk(k=self.k, dim=-1, largest=False)  # [B, N, k]
            dist_recip = 1. / (dists + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)
            # [B, N, k, C] -> [B, N, C]
            ret.feature = torch.sum(index_points(src.feature, idx) * weight.unsqueeze(-1), dim=-2)

        if dst.feature is not None:
            ret.feature = torch.cat([dst.feature, ret.feature], dim=-1)

        ret.feature = self.mlp(ret.feature.permute(0, 2, 1)).permute(0, 2, 1)
        ret.feature = self.drop(ret.feature)
        return ret
