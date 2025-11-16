# FROM: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import common


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # [B, S, N]
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = sqrdists.argmin(dim=-1, keepdim=True).repeat([1, 1, nsample])
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
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
            idx_prev=item
        )

    def as_input(self):
        args = self.xyz, self.feature
        return args[0] if args[1] is None else torch.concat(args, dim=-1)

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


def sample_and_group(latent, npoint, radius, nsample):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = latent.xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(latent.xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(latent.xyz, fps_idx)
    idx = query_ball_point(radius, nsample, latent.xyz, new_xyz)
    grouped_xyz = index_points(latent.xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if latent.feature is not None:
        grouped_points = index_points(latent.feature, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    return dict(xyz=new_xyz, feature=new_points, idx_prev=fps_idx)


def sample_and_group_all(latent):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    B, N, C = latent.xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(latent.xyz.device)
    grouped_xyz = latent.xyz.view(B, 1, N, C)

    if latent.feature is not None:
        new_points = torch.cat([grouped_xyz, latent.feature.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return dict(xyz=new_xyz, feature=new_points, idx_prev=None)


class PointNetSetAbstraction(nn.Module):

    def __init__(self, in_channel, npoint, radius, nsample, mlp):
        super().__init__()
        self.group_all = npoint == 1
        assert not (self.group_all and any((radius, nsample)))

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = common.ConvBnAct2d.create_mlp(in_channel, c2s=mlp)

    def extra_repr(self) -> str:
        attr = {"n2": self.npoint}
        if not self.group_all: attr.update({"r": self.radius, "k": self.nsample})
        return ", ".join(f"{k}={v}" for k, v in attr.items())

    def forward(self, latent: Latent):
        ret = sample_and_group_all(latent) if self.group_all \
            else sample_and_group(latent, self.npoint, self.radius, self.nsample)
        # feat: [B, N, K, C] -> [B, C, N, K] -> [B, C', N] -> [B, N, C']
        ret["feature"] = self.mlp(ret["feature"].permute(0, 3, 1, 2)).max(dim=-1)[0].permute(0, 2, 1)
        return Latent(**ret, latent_prev=latent)


class PointNetSetAbstractionMsg(nn.Module):

    def __init__(self, in_channel, npoint, radius_list, nsample_list, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_blocks = nn.ModuleList([
            common.ConvBnAct2d.create_mlp(in_channel + 3, c2s=mlp_list[i])
            for i in range(len(mlp_list))
        ])

    def extra_repr(self) -> str:
        attr = {"n2": self.npoint, "r": self.radius_list, "k": self.nsample_list}
        return ", ".join(f"{k}={v}" for k, v in attr.items())

    def forward(self, latent: Latent):
        B, N, *_ = latent.xyz.shape
        S = self.npoint

        fps = farthest_point_sample(latent.xyz, S)
        ret = dict(xyz=index_points(latent.xyz, fps), idx_prev=fps)

        new_points_list = []
        for i, (radius, K) in enumerate(zip(self.radius_list, self.nsample_list)):
            # [B, S, K, ...]
            group_latent = latent[query_ball_point(radius, K, latent.xyz, ret["xyz"])]
            group_latent.xyz -= ret["xyz"].view(B, S, 1, -1)

            grouped_points = group_latent.as_input().permute(0, 3, 1, 2)  # [B, C, S, K]
            grouped_points = self.mlp_blocks[i](grouped_points)

            new_points_list.append(grouped_points.max(dim=-1)[0])

        # feat: [B, C', N] -> [B, N, C']
        ret["feature"] = torch.cat(new_points_list, dim=1).permute(0, 2, 1)
        return Latent(**ret, latent_prev=latent)


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp = common.ConvBnAct1d.create_mlp(in_channel, c2s=mlp)

    def forward(self,
                latent1: Latent,
                latent2: Latent) -> Latent:
        """
        :param latent1: input points data
        :param latent2: sampled input points data
        :return: upsampled points data
        """
        B, N, C = latent1.xyz.shape
        _, S, _ = latent2.xyz.shape

        if S == 1:
            interpolated_points = latent2.feature.repeat(1, N, 1)
        else:
            dists = square_distance(latent1.xyz, latent2.xyz)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(latent2.feature, idx) * weight.view(B, N, 3, 1), dim=2)

        if latent1.feature is not None:
            new_points = torch.cat([latent1.feature, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.mlp(new_points)
        return Latent(xyz=latent1.xyz, feature=new_points.permute(0, 2, 1))
