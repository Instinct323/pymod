from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
# pip install grad-cam
from pytorch_grad_cam import EigenGradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from torch import nn
from tqdm import tqdm


class Target:

    def __init__(self, category=None):
        self.category = self.i = category

    def __call__(self, x):
        self.i = self.category
        if self.i is None:
            self.i = x.argmax(dim=-1).item()
        return x[..., self.i]


def grad_cam(model: nn.Module,
             tar_layers: list,
             images: Sequence[np.ndarray],
             targets=None,
             project=Path('gradcam'),
             gb=True,
             method=EigenGradCAM,
             use_cuda=False,
             mean_std=[[0., 0., 0.], [1., 1., 1.]], ):
    ''' Documentation: https://jacobgil.github.io/pytorch-gradcam-book
        e.g.,
        model = torchvision.models.resnet50(pretrained=True)
        target_layers = [model.layer4]'''
    model.eval(), project.mkdir(parents=True, exist_ok=True)
    mean_std = torch.tensor(mean_std)[..., None, None]
    imwrite = lambda file, img: cv2.imwrite(str(project / file), img)
    cam = method(model=model, target_layers=tar_layers, use_cuda=use_cuda)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    # 补全目标
    targets = targets if hasattr(targets, '__len__') else [targets]
    targets = [(tar if callable(tar) else Target(tar)) for tar in targets]
    targets = targets * len(images) if len(targets) == 1 else targets

    for i, (img, tar) in tqdm(enumerate(zip(images, targets)), total=len(images), desc='Grad-CAM'):
        bgr_img = img.astype(np.float32) / 255
        input_tensor = torch.from_numpy(np.ascontiguousarray(bgr_img[..., ::-1].transpose(2, 0, 1)))[None]
        # 图像标准化
        input_tensor = (input_tensor - mean_std[0]) / mean_std[1]
        # 绘制并保存 CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[tar])
        cam_image = show_cam_on_image(bgr_img, grayscale_cam[0], use_rgb=False)
        # 规定文件命名格式
        get_name = lambda x: f'{tar.i}-{i}-{x}.png'
        imwrite(get_name('cam'), cam_image)
        if gb:
            # Guided backprop
            gback = gb_model(input_tensor, target_category=tar.i)
            cam_mask = grayscale_cam[0, ..., None].repeat(3, -1)
            # 绘制并保存 GuidedBP, Grad-CAM
            imwrite(get_name('gb'), deprocess_image(gback))
            imwrite(get_name('camgb'), deprocess_image(cam_mask * gback))


if __name__ == '__main__':
    import torchvision

    image = map(lambda x: cv2.imread(str(x)),
                [Path('../data/both.png'), Path('../data/dog_cat.jfif')])
    model = torchvision.models.resnet50(pretrained=True)
    target_layers = [model.layer4]

    grad_cam(model, images=tuple(image), tar_layers=target_layers,
             project=Path('__pycache__'), targets=[0, 2],
             mean_std=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
