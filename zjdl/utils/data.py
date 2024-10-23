import copy
import multiprocessing as mp
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Callable, Sequence

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .bbox import *
from .imgtf import *
from pymod.utils.utils import LOGGER, Path


def ObjectArray(iterable):
    tmp = np.zeros(len(iterable), dtype=object)
    for i, obj in enumerate(iterable): tmp[i] = obj
    return tmp


def bp_times(loader: DataLoader, epochs: int):
    return len(loader) * epochs


def hold_out(cls_cnt: pd.DataFrame, scale: float, seed=0):
    """ :param cls_cnt: Dataframe[classes, img_id]
        :param scale: 各类别分布在训练集中的比例
        :return: 训练集 id 列表, 验证集 id 列表"""
    cls_cnt = cls_cnt.copy(deep=True)
    dtype = np.int64 if "int" in str(next(iter(cls_cnt.dtypes))) else np.float64
    radio = scale / (1 - scale)
    # 打乱图像的次序
    idx = cls_cnt.index.values
    np.random.seed(seed)
    np.random.shuffle(idx)
    # 记录训练集、验证集当前各类别数量
    data_cnt = np.zeros([2, len(cls_cnt.columns)], dtype=np.float64)
    data_cnt[1] += 1e-4
    # 存放训练集、验证集数据的 id
    data_pool = [[] for _ in range(2)]
    pbar = tqdm(idx)
    # 留出法: 计算期望比例 与 执行动作后比例的 SSE 损失
    loss_func = lambda x: np.square(x[0] / x[1] - radio).sum()
    for i in pbar:
        cnt = cls_cnt.loc[i]
        loss = np.zeros(2, dtype=np.float64)
        for j, next_sit in enumerate([data_cnt.copy() for _ in range(2)]):
            next_sit[j] += cnt
            loss[j] = loss_func(next_sit)
        # 根据损失值选择加入的数据集
        choose = loss.argmin()
        data_cnt[choose] += cnt
        data_pool[choose].append(i)
        # 输出当前的分割情况
        cur_scale = data_cnt[0] / data_cnt.sum(axis=0) - scale
        pbar.set_description(f"Category scale error ∈ [{cur_scale.min():.3f}, {cur_scale.max():.3f}]")
    # 输出训练集、验证集信息
    data_cnt = data_cnt.round(3).astype(dtype)
    LOGGER.info(f"Train Set ({len(data_pool[0])}): {data_cnt[0]}")
    LOGGER.info(f"Eval Set ({len(data_pool[1])}): {data_cnt[1]}")
    return data_pool


def undersampling(cls_cnt: pd.DataFrame, n, seed=0):
    """ :param cls_cnt: Dataframe[classes, img_id]
        :param n: 各类别实例的采样数量 (int, float, list, tuple)
        :return: 训练集 id 列表, 验证集 id 列表"""
    cls_cnt = cls_cnt.copy(deep=True)
    dtype = np.int64 if "int" in str(next(iter(cls_cnt.dtypes))) else np.float64
    np.random.seed(seed)
    cls_cnt_backup = cls_cnt
    n_cls = len(cls_cnt.columns)
    # 对参数 n 进行修改 / 校验
    if not hasattr(n, "__len__"): n = [n] * n_cls
    assert len(n) == n_cls, "The parameter n does not match the number of categories"
    # 筛选出无标签数据
    g = dict(list(cls_cnt.groupby(cls_cnt.sum(axis=1) == 0, sort=False)))
    unlabeled, cls_cnt = map(lambda k: g.get(k, pd.DataFrame()), (True, False))
    unlabeled = list(unlabeled.index)
    np.random.shuffle(unlabeled)
    # 存放训练集、验证集数据的 id
    m = len(unlabeled) // 2
    data_pool = [unlabeled[:m], unlabeled[m:]]
    data_cnt = np.zeros(n_cls, dtype=np.float64)
    while not cls_cnt.empty:
        # 取出当前 cls_cnt 最少的类
        j = cls_cnt.sum().apply(lambda x: np.inf if x == 0 else x).argmin()
        g = dict(list(cls_cnt.groupby(cls_cnt[j] > 0, sort=False)))
        # 对阳性样本进行划分, 放回阴性样本
        posi, cls_cnt = map(lambda k: g.get(k, pd.DataFrame()), (True, False))
        m, idx = -1, list(posi.index)
        if not posi.empty:
            lack = n[j] - data_cnt[j]
            if lack > 0:
                # 选取前 m 个加入训练集
                np.random.shuffle(idx)
                posi = posi.loc[idx]
                cumsum = np.cumsum(posi[j].to_numpy())
                m = np.abs(cumsum - lack).argmin() + 1
                # 考虑极端情况下, 不加入更好
                if m == 1 and cumsum[0] > lack: m = 0
                data_pool[0] += idx[:m]
                data_cnt += posi.iloc[:m].sum()
        # 其余放入验证集
        data_pool[1] += idx[m:]
    # 输出训练集、验证集信息
    data_cnt = data_cnt.to_numpy()
    LOGGER.info(f"Train Set ({len(data_pool[0])}): {data_cnt.round(3).astype(dtype)}")
    eval_cnt = cls_cnt_backup.sum().to_numpy() - data_cnt
    LOGGER.info(f"Eval Set ({len(data_pool[1])}): {eval_cnt.round(3).astype(dtype)}")
    return data_pool


class ImagePool:

    def __init__(self, files, labels):
        self.files = ObjectArray(files)
        self.labels = ObjectArray(labels)
        self.images = None
        self.img_size = None

    def select(self, indexes):
        self.files = self.files[indexes]
        self.labels = self.labels[indexes]
        if isinstance(self.images, np.ndarray):
            self.images = self.images[indexes]

    def loadimg(self,
                img_size: Union[int, tuple] = None,
                loader: Callable[[Path, tuple], np.ndarray] = load_img):
        if self.img_size != img_size:
            self.img_size = img_size
            LOGGER.info(f"Change the image size to {img_size}")
            # loader(file, img_size): 图像加载函数
            loader = partial(loader, img_size=img_size)
            # 启动多线程读取图像
            qbar = tqdm(ThreadPool(mp.cpu_count()).imap(loader, self.files),
                        total=len(self.files), desc="Loading images")
            self.images = ObjectArray(tuple(qbar))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


class _BaseDataset(Dataset):

    def __init__(self, imgpool, indexes=None):
        super().__init__()
        self.imgpool = imgpool
        self.indexes = indexes if hasattr(indexes, "__len__") else np.arange(len(imgpool))

    def __iter__(self):
        return (self[i] for i in self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __add__(self, other):
        _c = copy.copy(self)
        _c.indexes = self.indexes + other.indexes
        return _c

    def __iadd__(self, other):
        self.indexes += other.indexes
        return self


class SimpleDataset(_BaseDataset):
    """ :param indexes: 数据集的 ID 列表"""
    tf_types = [ColorJitter, RandomFlip, GaussianBlur]

    def __init__(self,
                 imgpool: ImagePool,
                 indexes: Sequence = None,
                 aughyp: dict = None):
        super().__init__(imgpool, indexes)
        self.aug = Transform(*(tf(aughyp) for tf in self.tf_types)) if aughyp else None

    def __getitem__(self, item):
        img, label = self.imgpool[self.indexes[item]]
        if self.aug: img = self.aug(img)
        return to_tensor(img), label


class PairDataset(SimpleDataset):

    def __getitem__(self, item):
        img, label = self.imgpool[self.indexes[item]]
        return to_tensor(img), to_tensor(self.aug(img)), label


class MosaicDataset(_BaseDataset):
    """ loads images in a 4-mosaic (generating square images)"""

    def __init__(self,
                 imgpool: ImagePool,
                 indexes: Sequence = None,
                 aughyp: dict = {},
                 labeltf=BBoxTransformer):
        super().__init__(imgpool, indexes)
        self.labeltf = labeltf
        self.flip = RandomFlip(aughyp)
        self.aug = Transform(ColorJitter(aughyp),
                             GaussianBlur(aughyp))
        # 获取 Mosaic 图像随机裁剪参数
        self.scale = aughyp.get("scale", .9)
        self.trans = aughyp.get("trans", .3)
        for p in (self.scale, self.trans): assert 0 <= p <= 1

    def __getitem__(self, item):
        """ 使用 self.labeltf 进行标签转换: affine -> aggregate -> flip"""
        s: int = self.imgpool.img_size
        # 随机选择 4 张图像
        item = [item] + np.random.choice(self.indexes, size=3).tolist()
        imgs, labels = map(list, zip(*(self.imgpool[i] for i in item)))
        # 生成随机裁剪的参数
        scale = np.random.uniform(-self.scale, self.scale) + 1
        x0, y0 = s * (.5 - scale + np.random.uniform(-self.trans, self.trans, 2))
        # 生成空白图像, 开始聚合
        img4 = np.full((s * 2, s * 2, imgs[0].shape[2]), BG_COLOR, dtype=np.uint8)
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            dh, dw = s - h, s - w
            # 根据图像位置指定填充尺寸
            row, col = i // 2, i % 2
            t, b, l, r = (row == 0) * dh, (row == 1) * dh, (col == 0) * dw, (col == 1) * dw
            img4[row * s: (row + 1) * s, col * s: (col + 1) * s] = \
                cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=(BG_COLOR,) * 3)
            # 对标签进行仿射变换
            labels[i] = self.labeltf.affine(img, labels[i], r=scale,
                                            x=x0 + (l, s)[col] * scale, y=y0 + (t, s)[row] * scale)
        labels = self.labeltf.aggregate(img4, labels)
        # 裁剪, 放缩
        img4 = cv2.warpAffine(img4, np.array([[scale, 0, x0], [0, scale, y0]]),
                              dsize=(s,) * 2, borderValue=(BG_COLOR,) * 3)
        # 翻转, 其它增强
        flip = self.flip.get_param()
        return self.flip.apply(self.aug(img4), **flip), self.labeltf.flip(img4, labels, **flip)


class CocoDetect:

    def __new__(cls, root, aughyp):
        cache_t = (root / "train.cache").lazy_obj(cls.make_index,
                                                  imgdir=root / "images/train2017",
                                                  labeldir=root / "labels/train2017")
        train = MosaicDataset(ImagePool(*cache_t), aughyp=aughyp.yaml())
        cache_v = (root / "val.cache").lazy_obj(cls.make_index,
                                                imgdir=root / "images/val2017",
                                                labeldir=root / "labels/val2017")
        val = MosaicDataset(ImagePool(*cache_v))
        return train, val

    @staticmethod
    def make_index(imgdir: Path,
                   labeldir: Path):
        img = imgdir.collect_file(formats=IMG_FORMAT)
        label = []
        for f in tqdm(img, "Loading labels"):
            f = labeldir / f"{f.stem}.txt"
            v = np.array(list(map(float, f.read_text().split()))).reshape(-1, 5)
            label.append(v)
        return img, label


if __name__ == "__main__":
    np.random.seed(0)

    # 5000 张图像各个类别边界框数目统计结果
    example = (np.random.random([100, 3]) * 3).astype(np.int32)
    example = pd.DataFrame(example, index=[f"train_{i}.jpg" for i in range(example.shape[0])])
    example *= np.array([4, 1, 9])

    for i in range(2):
        # train_id, eval_id = hold_out(example, 0.8)
        train_id, eval_id = undersampling(example, 50)
        print(example.loc[train_id].sum())
