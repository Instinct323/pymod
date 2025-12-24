import os
from typing import Callable

import numpy as np
import torch


def analyse_lora(model_id: str,
                 group_depth: int,
                 cmap: str = "gist_rainbow") -> dict:
    """
    :param model_id: model id
    :param group_depth: group depth
    :param cmap: color map
    :return: weight norm
    """
    import peft

    state_dict = peft.load_peft_weights(model_id, device="cpu")
    state_dict = simplify_state_dict(state_dict)

    g_i = group_idx(state_dict, depth=group_depth)
    state_dict = {k: state_dict[k] for k in sorted(state_dict, key=lambda k: g_i[k])}
    keys = sorted({k.rsplit(".", maxsplit=2)[0] for k in sorted(state_dict)}, key=lambda k: g_i[k + ".lora_A.weight"])

    # weight norm
    result = {}
    for k in keys:
        a = state_dict[k + ".lora_A.weight"]
        b = state_dict[k + ".lora_B.weight"]
        result[k] = torch.norm(b @ a).item()

    # plot
    index = np.array([g_i[k + ".lora_A.weight"] for k in result])
    color = plt.get_cmap(cmap)(index / index.max())
    plt.bar(range(len(result)), result.values(), color=color)
    return result


def group_idx(state_dict: dict,
              depth: int) -> dict:
    """
    :param state_dict: model state dict
    :param depth: group depth
    :return: group index
    """
    ret, sep = {}, "."
    for k in state_dict:
        idx = k.split(sep)[:depth]
        idx = tuple((int(i) if i.isdigit() else i) for i in idx)
        ret[k] = idx

    idx_unique = sorted(set(ret.values()))
    for k in ret:
        ret[k] = idx_unique.index(ret[k])
    return ret


def simplify_state_dict(state_dict: dict,
                        prefix: str = None,
                        filt: Callable = None) -> dict:
    """
    :param state_dict: model state dict
    :param prefix: prefix to remove
    :param filt: filter function
    :return: simplified state dict
    """
    if filt: state_dict = {k: v for k, v in state_dict.items() if filt(k)}
    len_prefix = len(os.path.commonprefix(list(state_dict)) if prefix is None else prefix)
    return {k[len_prefix:]: v for k, v in state_dict.items()}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path

    root = Path("/media/tongzj/Data/Workbench/Lab/ecg/runs/lightning_logs/4B-r8/checkpoints/epoch=13")
    ret = analyse_lora(root, group_depth=2)

    plt.grid()
    plt.show()
    # plt.xticks(range(len(ret)), ret.keys(), rotation=90)
