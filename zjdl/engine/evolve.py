import logging
import pickle
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml

from .result import Result

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def try_except(func):
    # try-except function. Usage: @try_except decorator
    @wraps(func)
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            LOGGER.error(f"{type(error).__name__}: {error}")

    return handler


# lower lim, upper lim, pace
META = {
    "unique": {"weight_decay": [0, 1, 1e-5],
               "fl_gamma": [0, 3, 1e-3],
               "hsv_h": [0, 0.1, 1e-2],
               "hsv_s": [0, 0.9, 1e-2],
               "hsv_v": [0, 0.9, 1e-2]},
    "general": {"kernel": [1, 15, 2],
                "width": [4, 2048, 4],
                "depth": [1, 10, 1],
                "float": [0, 1e10, 1e-8],
                "uint": [0, 1e10, 1],
                "proba": [0, 1, 1e-2]}
}


def collect_param(hyp):
    meta = {}
    # 专有超参数
    for key in filter(lambda k: k in hyp, META["unique"]):
        meta[key] = META["unique"][key]
    # 通用超参数
    for suffix in META["general"]:
        for key in filter(lambda k: k.endswith(f"_{suffix}"), hyp):
            meta[key] = META["general"][suffix]
    return meta


class InertiaOpt:

    def set_meta(self):
        default_state = [self.patience, 0]
        # meta: lower lim, upper lim, pace, patience, greedy momentum
        self.meta = {k: v + default_state for k, v in collect_param(self.hyp).items()}

    def __init__(self, project, hyp, patience=5):
        self.project = project
        self.project.mkdir(parents=True, exist_ok=True)
        self.title = "evolve", "hyp", "previous", "current", "momentum", "better"
        self.result = Result(self.project, title=self.title[1:-1] + ("fitness",))
        # 读取 yaml 文件并设置为属性
        if not isinstance(hyp, dict):
            hyp = yaml.load(hyp.read_text(), Loader=yaml.Loader)
        self.hyp = hyp
        # 搜索过程变量
        self.patience = patience
        self.set_meta()
        self.seed = round(time.time())

    def save_or_load(self, save=True):
        hyp_f = self.project / "hyp.yaml"
        state_f = self.project / "evolve.yaml"
        # 保存状态信息
        if save:
            self.hyp.update({"__seed": self.seed})
            hyp_f.write_text(yaml.dump(self.hyp))
            state_f.write_text(yaml.dump(self.meta))
        # 加载状态信息
        else:
            if state_f.is_file():
                self.meta = yaml.load(state_f.read_text(), Loader=yaml.Loader)
            if hyp_f.is_file():
                self.hyp = yaml.load(hyp_f.read_text(), Loader=yaml.Loader)
                for key in filter(lambda k: k.startswith("__"), tuple(self.hyp)):
                    setattr(self, key[2:], self.hyp.pop(key))

    def __call__(self, fitness, epochs, mutation=.5):
        """ :param fitness(hyp, epoch) -> float: 适应度函数
            :param epochs: 超参数进化总轮次
            :param mutation: 基因突变时的浮动百分比"""
        epoch, best_fit = len(self.result) - 1, self.result["fitness"].max()
        self.save_or_load(save=False)
        LOGGER.info(f"Evolving hyperparameters: {list(self.meta)}")
        # 获取 baseline 的 fitness
        if epoch == -1:
            epoch += 1
            best_fit = fitness(self.hyp, 0)
            LOGGER.info(f"The fitness of baseline: {best_fit:.4g}\n")
            self.result.record(("baseline", 0, 0, 0, best_fit), epoch)
        # 保存当前状态信息
        self.save_or_load(save=True)
        while epoch < epochs and self.meta:
            # 设置随机数种子, 确保训练中断时可恢复
            np.random.seed(self.seed)
            self.seed += 1
            # 随机选取一个超参数
            p = np.array([self.meta[key][-2] for key in self.meta]) ** 2
            key = np.random.choice(tuple(self.meta), p=p / p.sum())
            # 根据当前值进行搜索
            lower_lim, upper_lim, pace, patience, m_greed = self.meta[key]
            range_ = round((upper_lim - lower_lim) / pace)
            previous = round((self.hyp[key] - lower_lim) / pace)
            # 根据是否到达上 / 下限确定动量
            m_forced = {0: 1, range_: -1}.get(previous, 0)
            momentum = np.random.choice((-1, 1)) if not (m_greed or m_forced) else np.sign(sum((m_greed, m_forced)))
            # m_greed, m_forced 符号相反时不再搜索
            skip = not momentum
            if not skip:
                epoch += 1
                current = np.clip(previous + momentum *
                                  np.random.randint(1, max(1, round(previous * mutation)) + 1),
                                  a_min=0, a_max=range_).item()
                momentum = round(np.clip((current - previous) / (previous + 1e-8), a_min=-1, a_max=1).item(), 3)
                # 修改超参数字典
                hyp = self.hyp.copy()
                hyp[key] = current * pace + lower_lim
                LOGGER.info(f"\n{__class__.__name__} epoch {epoch}: "
                            f"Change <{key}> from {self.hyp[key]:.4g} to {hyp[key]:.4g}\n")
                # 计算适应度
                fit = fitness(hyp, epoch)
                better = fit - best_fit
                LOGGER.info(("\n" + " %9s" * 6) % self.title)
                LOGGER.info((" %9s" * 2 + " %9.4g" * 4) % \
                            (f"{epoch}/{epochs}", key[:9], self.hyp[key], hyp[key], momentum, better) + "\n")
                self.result.record((key, self.hyp[key], hyp[key], momentum, fit), epoch)
                # 保存状态信息
                patience += 2 if better > 0 else -1
                self.meta[key][-2:] = min(2 * self.patience, patience), int(np.sign(momentum * better).item())
                skip = patience <= 0
                if better > 0: self.hyp, best_fit = hyp, fit
            # 结束该超参数的搜索
            if skip: self.meta.pop(key)
            self.save_or_load(save=True)
        best = self.result["fitness"].argmax()
        LOGGER.info(f"Hyperparameter evolution is complete, the best epoch is {best}.\n")
        self.plot(show=False)
        return self.hyp

    @try_except
    def plot(self, show=True):
        # 对各个超参数进行分类, 记录最优适应度曲线
        hyp, best = {}, [(0, self.result.loc[0, "fitness"])]
        for i in range(1, len(self.result)):
            r = self.result.iloc[i]
            p = i, r["fitness"]
            hyp.setdefault(r["hyp"], []).append(p + (r["momentum"],))
            if p[1] >= best[-1][1]: best.append(p)
        # 绘制最优适应度曲线
        best = np.array(best).T
        for i in "xy": getattr(plt, f"{i}ticks")([], [])
        plt.plot(*best, color="gray", alpha=.5, linestyle="--")
        # 绘制各个超参数的散点
        for k in sorted(hyp):
            v = np.array(hyp[k]).T
            alpha = (v[-1] + 1) * .35 + .3
            plt.scatter(*v[:2], alpha=alpha, label=k)
        # 根据最优适应度曲线设置上下限
        # plt.ylim((best[1, 0], best[1, -1]))
        plt.legend(frameon=True)
        plt.savefig(self.project / "curve.png")
        plt.show() if show else plt.close()


class BayesOpt:

    def __init__(self, project, hyp):
        self.project = project
        self.project.mkdir(parents=True, exist_ok=True)
        # 读取 yaml 文件并设置为属性
        if not isinstance(hyp, dict):
            hyp = yaml.load(hyp.read_text(), Loader=yaml.Loader)
        self.hyp = hyp
        # 设置其它属性
        self.meta = collect_param(hyp)
        self.seed = round(time.time())
        self.study = optuna.create_study(study_name="Evolving hyperparameters", direction="maximize")

    def save_or_load(self, save=True):
        hyp_f = self.project / "hyp.yaml"
        state_f = self.project / "evolve.cache"
        # 保存状态信息
        if save:
            self.hyp.update({"__seed": self.seed})
            hyp_f.write_text(yaml.dump(self.hyp))
            state_f.write_bytes(pickle.dumps(self.study.get_trials(deepcopy=False)))
        # 加载状态信息
        else:
            if state_f.is_file():
                self.study.add_trials(pickle.loads(state_f.read_bytes()))
            if hyp_f.is_file():
                self.hyp = yaml.load(hyp_f.read_text(), Loader=yaml.Loader)
                for key in filter(lambda k: k.startswith("__"), tuple(self.hyp)):
                    setattr(self, key[2:], self.hyp.pop(key))

    def __call__(self, fitness, epochs, n_suggest=3):
        """ :param fitness(hyp, epoch) -> float: 适应度函数
            :param epochs: 超参数进化总轮次"""
        n_suggest = min(n_suggest, len(self.hyp))
        self.save_or_load(save=False)
        # 检查最后一个 trials 是否未完成
        trials = self.study.get_trials(deepcopy=False)
        is_running = lambda: len(trials) and not trials[-1].state
        for epoch in range(len(trials) - is_running(), epochs):
            # 设置随机数种子, 确保训练中断时可恢复
            np.random.seed(self.seed)
            # 根据最后一个 trial 的状态获取 trial
            trials = self.study.get_trials(deepcopy=False)
            tr = trials[-1] if is_running() else self.study.ask()
            # 获取建议值, 并保存 trial
            for k in np.random.choice(tuple(self.meta), n_suggest, replace=False):
                low, high, step = self.meta[k]
                getattr(tr, f"suggest_{type(step).__name__}")(name=k, low=low, high=high, step=step)
            self.save_or_load(save=True)
            LOGGER.info(f"\n{__class__.__name__} epoch {epoch}: {tr.params}")
            # 创建新的超参数字典
            hyp = self.hyp.copy()
            hyp.update(tr.params)
            # 更新最优超参数
            self.study.tell(epoch, fitness(hyp, epoch=tr.number))
            self.hyp.update(self.study.best_params)
            self.seed += 1
            self.save_or_load(save=True)
        return self.hyp

    @try_except
    def plot(self):
        optuna.visualization.plot_optimization_history(self.study).show()


if __name__ == "__main__":
    from pathlib import Path


    def fitness(hyp, epoch=None):
        loss = 0
        for key, (low_lim, up_lim, pace) in evolve.meta.items():
            if key in hyp:
                loss += (hyp[key] / up_lim - 0.5) ** 2
        return - loss


    evolve = BayesOpt(Path("__pycache__"), Path("cfg/hyp.yaml"))
    print(evolve(fitness, 300))
