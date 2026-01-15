import torch


def info_batch(batch) -> dict:
    if isinstance(batch, dict):
        kv_iter = batch.items()
    elif isinstance(batch, (tuple, list)):
        kv_iter = enumerate(batch)
    else:
        raise TypeError(f"{type(batch)} is not supported")

    ret = {}
    for k, v in kv_iter:
        ret[k] = {"type": type(v)}
        if isinstance(v, torch.Tensor):
            ret[k].update({"shape": v.shape, "dtype": v.dtype})
    for k, v in ret.items():
        print(f"{k}: {v}")
    return ret


def is_main_process() -> bool:
    import os
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def wait_for_cuda(sleep: float = 10.):
    if not is_main_process(): return
    import nvitop, time

    devices = nvitop.Device.all()
    while True:
        wait = False
        for dev in devices:
            for process in dev.processes().values():
                wait |= process.command().startswith("python")

        if not wait: return
        print(f"Waiting for CUDA devices to be free...")
        time.sleep(sleep)
