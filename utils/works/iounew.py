import torch
from torch import nn


class IouCalculator(nn.Module):
    ''' n: Number of batches per training epoch
        t: The epoch when mAP's ascension slowed significantly
        monotonous: {
            None: origin
            True: monotonic FM
            False: non-monotonic FM
        }
        pred, target: x0,y0,x1,y1'''

    def __init__(self, n=890, t=34, monotonous=False):
        super().__init__()
        time_to_real = n * t
        self.momentum = 1 - pow(0.05, 1 / time_to_real)
        self.iou_mean = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.monotonous = monotonous

    def update(self, vspace):
        if self.training:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * vspace.iou.detach().mean())

    def _scaled_loss(self, loss, vspace, alpha=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            beta = vspace.iou.detach() / self.iou_mean
            if vspace.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = delta * torch.pow(alpha, beta - delta)
                loss *= beta / divisor
        return loss

    def __call__(self, pred, target):
        return self.VarSpace(pred, target)

    class VarSpace:

        def __getattr__(self, item):
            if callable(self._fget[item]):
                self._fget[item] = self._fget[item]()
            return self._fget[item]

        def __init__(self, pred, target):
            self.pred, self.target = pred, target
            self._fget = {
                # x,y,w,h
                'pred_xy': lambda: (self.pred[..., :2] + self.pred[..., 2: 4]) / 2,
                'pred_wh': lambda: self.pred[..., 2: 4] - self.pred[..., :2],
                'target_xy': lambda: (self.target[..., :2] + self.target[..., 2: 4]) / 2,
                'target_wh': lambda: self.target[..., 2: 4] - self.target[..., :2],
                # x0,y0,x1,y1
                'min_coord': lambda: torch.minimum(self.pred[..., :4], self.target[..., :4]),
                'max_coord': lambda: torch.maximum(self.pred[..., :4], self.target[..., :4]),
                # The overlapping region
                'wh_inter': lambda: torch.relu(self.min_coord[..., 2: 4] - self.max_coord[..., :2]),
                's_inter': lambda: torch.prod(self.wh_inter, dim=-1),
                # The area covered
                's_union': lambda: torch.prod(self.pred_wh, dim=-1) +
                                   torch.prod(self.target_wh, dim=-1) - self.s_inter,
                # The smallest enclosing box
                'wh_box': lambda: self.max_coord[..., 2: 4] - self.min_coord[..., :2],
                's_box': lambda: torch.prod(self.wh_box, dim=-1),
                'l2_box': lambda: torch.square(self.wh_box).sum(dim=-1),
                # The central points' connection of the bounding boxes
                'd_center': lambda: self.pred_xy - self.target_xy,
                'l2_center': lambda: torch.square(self.d_center).sum(dim=-1),
                # IoU
                'iou': lambda: 1 - self.s_inter / self.s_union
            }
