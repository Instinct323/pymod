import torch


class ItemStack:

    def __init__(self):
        self.data = None

    def __bool__(self):
        return self.data is not None

    def pop(self,
            all_gather_func=None):
        if all_gather_func:
            shape = self.data.shape[1:]
            self.data = all_gather_func(self.data).view(-1, *shape)
        data, self.data = self.data, None
        return data

    def push(self, items: torch.Tensor):
        self.data = items if not self else torch.cat([self.data, items], dim=0)
