import bisect
from typing import Iterable, Dict

import numpy as np
from scipy.spatial import transform

SO3 = transform.Rotation  # 特殊正交群


class Translation:
    """ 位移向量"""

    def __init__(self, tx, ty, tz):
        self.t = np.array([tx, ty, tz])

    def as_matrix(self):
        return self.t[:, None]

    def __neg__(self):
        return Translation(*(-self.t))


class SE3:
    """ 特殊欧式群"""

    def __init__(self, R: SO3, t: Translation):
        assert isinstance(R, SO3) and isinstance(t, Translation)
        self.R = R
        self.t = t

    def as_matrix(self):
        return np.block([[self.R.as_matrix(), self.t.t[:, None]], [0, 0, 0, 1]])

    def as_vec7(self):
        return np.concatenate([self.t.t, self.R.as_quat()])

    def inverse(self):
        return SE3(self.R.inv(), Translation(*(-self.R.as_matrix() @ self.t.t)))

    def __mul__(self, other):
        if isinstance(other, SE3):
            return SE3(self.R * other.R, self * other.t)
        elif isinstance(other, Translation):
            return Translation(*(self.R.as_matrix() @ other.t + self.t.t))
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @classmethod
    def from_file(cls, file):
        with open(str(file), "r") as f:
            for line in f:
                line = line.replace("\t", " ").replace(",", " ").strip()
                if not line or line.startswith("#"): continue
                line = list(map(float, line.split()))
                # timestamp, tx, ty, tz
                if len(line) in (3, 4):
                    pose = np.array(line[-3:])
                    yield pose if len(line) & 1 else (int(line[0]), pose)
                # timestamp, tx, ty, tz, qw, qx, qy, qz
                if len(line) in (7, 8):
                    pose = cls(SO3.from_quat(np.append(line[-3:], line[-4])), Translation(*line[-7: -4]))
                    yield pose if len(line) & 1 else (int(line[0]), pose)
                # timestamp, 4x4 matrix
                elif len(line) in (12, 13, 16, 17):
                    m = np.array(line[len(line) & 1:]).reshape(-1, 4)
                    pose = cls(SO3.from_matrix(m[:3, :3]), Translation(*m[:3, 3]))
                    yield (int(line[0]), pose) if len(line) & 1 else pose
                else:
                    raise NotImplementedError(f"Unsupported format: {line}")


def nearest_timestamp(query: Iterable[int], value: Iterable[int], max_ts_diff: float = float("inf")):
    """ 最邻近时间戳"""
    cnt = 0
    query, value = map(sorted, (query, value))
    for q in query:
        lo = bisect.bisect_right(value, q)
        candidate = np.array(value[max(0, lo - 1): lo + 1])
        v = candidate[np.abs(q - candidate).argmin()]
        if (q - v) <= max_ts_diff:
            cnt += 1
            yield q, v
    print(f"Matching success rate: {cnt / len(query):.2%}")


def align_trajectory(pts1: np.ndarray, pts2: np.ndarray):
    """ Closed-form solution of absolute orientation using unit quaternions """
    assert pts1.ndim == 2 and pts1.shape[1] == 3 and pts1.shape == pts2.shape
    pts1, pts2 = map(np.float64, (pts1, pts2))
    # 尺度因子
    centroid1, centroid2 = pts1.mean(axis=0), pts2.mean(axis=0)
    pts1, pts2 = pts1 - centroid1, pts2 - centroid2
    s = np.sqrt(np.square(pts2).sum() / np.square(pts1).sum())
    # 单位四元数
    Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz = (pts1.T @ pts2).flatten()
    sumn = np.array([[(Sxx + Syy + Szz) / 2, Syz - Szy, Szx - Sxz, Sxy - Syx],
                     [0, (Sxx - Syy - Szz) / 2, Syx + Sxy, Szx + Sxz],
                     [0, 0, (- Sxx + Syy - Szz) / 2, Szy + Syz],
                     [0, 0, 0, (- Sxx - Syy + Szz) / 2]])
    sumn += sumn.T
    eig_val, eig_vec = np.linalg.eig(sumn)
    w, x, y, z = eig_vec[:, eig_val.argmax()]
    # 代入数值
    R = SO3.from_matrix(
        [[w ** 2 + x ** 2 - y ** 2 - z ** 2, -2 * w * z + 2 * x * y, 2 * w * y + 2 * x * z],
         [2 * w * z + 2 * x * y, w ** 2 - x ** 2 + y ** 2 - z ** 2, -2 * w * x + 2 * y * z],
         [-2 * w * y + 2 * x * z, 2 * w * x + 2 * y * z, w ** 2 - x ** 2 - y ** 2 + z ** 2]]
    )
    t = centroid2 - s * R.as_matrix() @ centroid1
    return s, R, t


def abs_trans_error(pred: np.ndarray, gt: np.ndarray):
    """ 绝对位移误差 (RMSE)"""
    assert pred.ndim == 2 and pred.shape[1] == 3 and pred.shape == gt.shape
    return np.sqrt(np.square(pred - gt).sum(axis=-1).mean())


def eval_trajectory(pred: Dict[int, SE3], gt: Dict[int, SE3], plot: bool = True):
    # 对齐时间戳
    pred_se, gt_se = [], []
    for tp, tgt in nearest_timestamp(pred.keys(), gt.keys()):
        pred_se.append(pred[tp])
        gt_se.append(gt[tgt])
    # 对齐轨迹
    pts1, pts2 = np.stack([x.t.t for x in pred_se]), np.stack([x.t.t for x in gt_se])
    # pts2 = (2.5 * SO3.from_quat([0.5, 0.1, 0.2, 0.3]).as_matrix() @ pts2.T).T + np.random.uniform(0, 0.2, pts2.shape)
    s, R, t = align_trajectory(pts1, pts2)
    pts1 = (s * R.as_matrix() @ pts1.T).T + t
    # 绘制轨迹
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.subplot(projection="3d")
        fig.plot(*pts1.T, "deepskyblue", label="pred")
        fig.plot(*pts2.T, "orange", label="GT")
        plt.show()
    return abs_trans_error(pts1, pts2)


if __name__ == "__main__":
    import os

    np.set_printoptions(3, suppress=True)

    os.chdir(r"D:\Workbench\data\dataset-room4_512_16\dso")

    ate = eval_trajectory("gt_imu.csv", "gt_imu.csv")
    print(ate)
