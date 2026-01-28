import torch
import torch.nn.functional as F


class ItemStack:

    def __init__(self):
        self.data = None

    def __bool__(self):
        return self.data is not None

    def pop(self,
            all_gather_func=None) -> torch.Tensor:
        if all_gather_func:
            shape = self.data.shape[1:]
            self.data = all_gather_func(self.data).view(-1, *shape)
        data, self.data = self.data, None
        return data

    def push(self, items: torch.Tensor):
        self.data = items if not self else torch.cat([self.data, items], dim=0)


class LogitsProcessor:

    def __init__(self,
                 mode: str):
        self.mode = ["binary", "multiclass", "multilabel"].index(mode)

    def __call__(self,
                 logits: torch.Tensor,
                 target: torch.Tensor = None,
                 dim: int = -1) -> dict:
        if self.mode == 0: logits = logits.squeeze(dim)
        ret = {
            "logits": logits,
            "score": (torch.softmax(logits, dim=dim) if self.mode == 1 else torch.sigmoid(logits)).detach(),
            "pred": (logits.argmax(dim) if self.mode == 1 else (logits > 0).long()).detach()
        }
        if target is not None:
            target = target.float() if self.mode == 0 else target
            ret["loss"] = (F.cross_entropy if self.mode == 1 else F.binary_cross_entropy_with_logits)(logits, target)
        return ret


if __name__ == '__main__':
    lp = LogitsProcessor("multiclass")
    x = torch.randn(3, 4)
    print(lp(x))
