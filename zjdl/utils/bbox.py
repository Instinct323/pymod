import cv2
import numpy as np


def xywh2xyxy(labels):
    labels = labels.copy()
    labels[..., 1:3] -= labels[..., 3:5] / 2
    labels[..., 3:5] += labels[..., 1:3]
    return labels


def xyxy2xywh(labels):
    labels = labels.copy()
    labels[..., 3:5] -= labels[..., 1:3]
    labels[..., 1:3] += labels[..., 3:5] / 2
    return labels


def pixel2radio(labels, h, w):
    labels = labels.copy()
    labels[..., 1::2] /= w
    labels[..., 2::2] /= h
    return labels


def radio2pixel(labels, h, w):
    labels = labels.copy()
    labels[..., 1::2] *= w
    labels[..., 2::2] *= h
    return labels


class BBoxTransformer:
    ''' labels: [cls, x1, y1, x2, y2]
        针对各种数据增强手段提供的标签变换'''

    @staticmethod
    def affine(img, label, r, x, y):
        ''' 在原图像截取左上顶点 (x, y) 的图像, 并以比例 r 进行放缩'''
        h, w = img.shape[:2]
        label = radio2pixel(label, h, w)
        label[..., 1:] *= r
        # bbox: [cls, x1, y1, x2, y2]
        label[..., 1::2] += x
        label[..., 2::2] += y
        return label

    @staticmethod
    def aggregate(img, label):
        ''' 多个标签聚合'''
        h, w = img.shape[:2]
        label = np.concatenate(label, axis=0)
        label[..., 1::2] = label[..., 1::2].clip(min=0, max=w - 1)
        label[..., 2::2] = label[..., 2::2].clip(min=0, max=h - 1)
        # 筛选掉过小的边界框
        return label[np.prod(label[..., 3:5] - label[..., 1:3], axis=-1) > 3]

    @staticmethod
    def flip(img, label, hflip, vflip):
        h, w = img.shape[:2]
        if hflip:
            label[..., 1::2] = w - 1 - label[..., 1::2]
        if vflip:
            label[..., 2::2] = h - 1 - label[..., 2::2]
        return label


class BBoxPlotter(list):

    def __init__(self, labels):
        super().__init__(map(str, labels))
        self.colors = np.random.randint(0, 255, [len(self), 3], dtype=np.uint8).tolist()

    def __call__(self, img, labels):
        img = img.copy()
        tl = max(1, round(0.003 * min(img.shape[:2])))
        has_p = len(labels) == 6
        # labels: [cls, x1, y1, x2, y2, *p]
        for cls, *xyxy in labels:
            cls = int(cls)
            # 获取边界框顶点, 绘制矩形
            c1, c2 = tuple(map(round, xyxy[:2])), tuple(map(round, xyxy[2:4]))
            cv2.rectangle(img, c1, c2, self.colors[cls], thickness=tl, lineType=cv2.LINE_AA)
            # 制作并绘制标签
            tag = self[cls] + (f' {xyxy[-1]:.2f}' if has_p else '')
            tf = max(1, tl - 1)  # font thickness
            t_size = cv2.getTextSize(tag, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, self.colors[cls], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, tag, (c1[0], c1[1] - 2), 0, tl / 3, (255,) * 3, thickness=tf, lineType=cv2.LINE_AA)
        return img
