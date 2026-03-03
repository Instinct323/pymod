import torch


def gpu_time(func, *args, **kwargs) -> float:
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    t0.record()
    func(*args, **kwargs)
    t1.record()

    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / 1e3


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


def wait_for_cuda(devices: list[int] = None,
                  sleep: float = 10.,
                  re_pattern: str = "python"):
    if not is_main_process(): return
    import nvitop, re, time

    devices = nvitop.Device.all() if devices is None else nvitop.Device.from_indices(devices)
    while True:
        wait = False
        for dev in devices:
            for process in dev.processes().values():
                wait |= bool(re.search(re_pattern, process.command()))

        if not wait: return
        print(f"Waiting for CUDA devices to be free...")
        time.sleep(sleep)
