import logging
import math
import os
from pathlib import Path

import torch
import yaml
from torch.cuda import amp

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def select_device(device="", batch_size=None, verbose=True):
    """ device: "cpu" or "0" or "0,1,2,3" """
    # 判断 GPU 可用状态, 设置环境变量
    cuda = device.lower() != "cpu" and torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = device if cuda else "-1"
    s = "Available devices: "
    if cuda:
        # 检查 batch_size 与 GPU 数量是否匹配
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        # 读取各个设备的信息并输出
        space = " " * len(s)
        for i, dev in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{dev} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"
    if verbose: LOGGER.info(s)
    return torch.device("cuda:0" if cuda else "cpu")


class CosineLR(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, lrf, epochs):
        lr_lambda = lambda x: lrf + (1 + math.cos(math.pi * x / epochs)) / 2 * (1 - lrf)
        super().__init__(optimizer, lr_lambda)


class Trainer:
    """ :param model: 网络模型
        :param project: 项目目录 (Path)
            :ivar best.pt: 最优模型的字典
            :ivar last.pt: 最新模型的字典
        :param hyp: 超参数字典
            :key epochs: 训练总轮次
            :key optimizer: 优化器类型
            :key lr0, lrf: 起始学习率, 最终学习率 (比例)
            :key weight_decay: 权值的 L2 范数系数
            :key device: 设备 id
            :key batch_size: 批尺寸"""
    training = property(lambda self: self.model.training)
    lr = property(lambda self: self._optim.param_groups[0]["lr"])

    def __init__(self, model, project, hyp):
        self.project = project
        self.project.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Logging results to {self.project}")
        # 优先读取项目下的 hyp.yaml
        hyp_file = self.project / "hyp.yaml"
        if hyp_file.is_file():
            hyp = yaml.load(hyp_file.read_text(), Loader=yaml.Loader)
        else:
            if isinstance(hyp, Path):
                hyp = yaml.load(hyp.read_text(), Loader=yaml.Loader)
            # 设置超参数字典的默认参数, 存储到项目目录
            hyp.setdefault("device", "")
            hyp.setdefault("optimizer", "SGD")
            hyp.setdefault("weight_decay", 0.)
            hyp_file.write_text(yaml.dump(hyp))
        self.hyp = hyp
        # 如果是 YamlModel 类型, 保存模型的配置文件
        cfg = getattr(model, "cfg", None)
        if isinstance(cfg, dict): (self.project / "cfg.yaml").write_text(yaml.dump(cfg))
        # 根据设备对模型进行设置
        self.device = select_device(hyp["device"], batch_size=hyp.get("batch_size", None))
        self.model = model.to(self.device)
        # 实例化优化器, lr 监听器
        self._epochs = hyp["epochs"]
        self._optim = getattr(torch.optim, hyp["optimizer"])(self.model.parameters(), lr=hyp["lr0"],
                                                             weight_decay=hyp["weight_decay"])
        self._lr_scheduler = CosineLR(self._optim, lrf=hyp["lrf"], epochs=self._epochs)
        self._scaler = amp.GradScaler(enabled=self.device.type != "cpu")
        # 加载最新的模型参数
        ckpt = self.load_ckpt("last.pt")
        self._cur_epoch = ckpt.get("epoch", -1) + 1
        torch.cuda.empty_cache()

    def __iter__(self):
        def generator():
            while self._cur_epoch < self._epochs:
                yield self._cur_epoch
                self._cur_epoch += 1

        return generator()

    @staticmethod
    def cuda_memory(divisor=1e9) -> float:
        return torch.cuda.memory_reserved() / divisor

    def load_ckpt(self, file: str = "last.pt") -> dict:
        file = self.project / file
        if not file.is_file(): return {}
        # 若文件存在, 则加载 checkpoint
        ckpt = torch.load(file, map_location=self.device)
        self._optim.load_state_dict(ckpt["optim"])
        self._lr_scheduler.load_state_dict(ckpt["sche"])
        self.model.load_state_dict(ckpt["model"], strict=True)
        return ckpt

    def save_ckpt(self, files: list = ["last.pt"], **ckpt_kwd):
        ckpt_kwd.update({"epoch": self._cur_epoch,
                         "optim": self._optim.state_dict(),
                         "sche": self._lr_scheduler.state_dict(),
                         "model": self.model.state_dict()})
        for f in files: torch.save(ckpt_kwd, self.project / f)

    def bp_gradient(self, loss) -> bool:
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optim)
        self._scaler.update()
        self._optim.zero_grad()
        return torch.isfinite(loss).item()
