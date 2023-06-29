import logging
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda import amp
from tqdm import tqdm

from .crosstab import Crosstab
from .result import Result

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def fstring(*args, length=9, decimals=4):
    lut = {'s': f'%{length}s', 'g': f'%{length}.{decimals}g'}
    # e.g., 'ssgg' -> '%9s %9s %9.4g %9.4g'
    fstr = ' '.join(lut['g' if isinstance(i, (int, float)) else 's'] for i in args)
    return fstr % args


def select_device(device='', batch_size=None, verbose=True):
    ''' device: "cpu" or "0" or "0,1,2,3" '''
    # 判断 GPU 可用状态, 设置环境变量
    cuda = device.lower() != 'cpu' and torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = device if cuda else '-1'
    s = 'Available devices: '
    if cuda:
        # 检查 batch_size 与 GPU 数量是否匹配
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        # 读取各个设备的信息并输出
        space = ' ' * len(s)
        for i, dev in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{dev} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    if verbose: LOGGER.info(s)
    return torch.device('cuda:0' if cuda else 'cpu')


class CosineLR(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, lrf, epochs, verbose=False):
        lr_lambda = lambda x: lrf + (1 + math.cos(math.pi * x / epochs)) / 2 * (1 - lrf)
        super().__init__(optimizer, lr_lambda, verbose=verbose)


class Trainer:
    ''' model: 网络模型
        project: 项目目录 (Path)
            best.pt: 最优模型的字典
            last.pt: 最新模型的字典
            result.json: 训练过程信息
        m_title: 实例方法 <metrics> 中各个度量的名称
        hyp: 超参数字典
            epochs: 训练总轮次
            lr0, lrf: 起始、最终学习率
            weight_decay: 权值的 L2 范数系数
        [WARN] Needs to be rewriten:
            loss -> tensor: train 调用, 返回带梯度的标量损失
            metrics -> numpy: eval 调用, 返回多个指标分数 (shape=[n,])
            fitness -> float: 根据 metrics 函数的返回值计算适应度'''
    training = property(fget=lambda self: self.model.training)

    def __init__(self, model, project, m_title, hyp):
        self.project = project
        self.project.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f'Logging results to {self.project}')
        assert len(m_title) > 0
        self._m_title = m_title
        # 优先读取项目下的 hyp.yaml
        hyp_file = self.project / 'hyp.yaml'
        if hyp_file.is_file():
            hyp = yaml.load(hyp_file.read_text(), Loader=yaml.Loader)
        else:
            if isinstance(hyp, Path):
                hyp = yaml.load(hyp.read_text(), Loader=yaml.Loader)
            # 设置超参数字典的默认参数, 存储到项目目录
            hyp.setdefault('device', '')
            hyp.setdefault('optimizer', 'SGD')
            hyp.setdefault('weight_decay', 0.)
            hyp_file.write_text(yaml.dump(hyp))
        self.hyp = hyp
        # 如果是 YamlModel 类型, 保存模型的配置文件
        cfg = getattr(model, 'cfg', None)
        if isinstance(cfg, dict): (self.project / 'cfg.yaml').write_text(yaml.dump(cfg))
        # 根据设备对模型进行设置
        self.device = select_device(hyp['device'], batch_size=hyp.get('batch_size', None))
        cuda = self.device.type != 'cpu'
        self.model = model.to(self.device)
        # 加载最优模型的信息
        self.best_fitness = - float('inf')
        model_file = self.project / 'best.pt'
        if model_file.is_file():
            metrics = torch.load(model_file, map_location=self.device)['metrics']
            if metrics.size > 0: self.best_fitness = self.fitness(metrics)
        # 实例化优化器, lr 监听器
        self._epochs = hyp['epochs']
        self._optim = getattr(torch.optim, hyp['optimizer'])(self.model.parameters(), lr=hyp['lr0'],
                                                             weight_decay=hyp['weight_decay'])
        self._lr_scheduler = CosineLR(self._optim, lrf=hyp['lrf'], epochs=self._epochs)
        self._scaler = amp.GradScaler(enabled=cuda)
        torch.cuda.empty_cache()

    def cuda_memory(self, divisor=1e9):
        return torch.cuda.memory_reserved() / divisor

    def load_ckpt(self, _file: str = 'last.pt') -> dict:
        _file = self.project / _file
        if not _file.is_file(): return {}
        # 若文件存在, 则加载 checkpoint
        ckpt = torch.load(_file, map_location=self.device)
        self._optim.load_state_dict(ckpt['optim'])
        self._lr_scheduler.load_state_dict(ckpt['sche'])
        self.model.load_state_dict(ckpt['model'], strict=True)
        return ckpt

    def save_ckpt(self, _files: list = ['last.pt'], **ckpt_kwd):
        ckpt_kwd.update({'optim': self._optim.state_dict(),
                         'sche': self._lr_scheduler.state_dict(),
                         'model': self.model.state_dict()})
        # 保存 checkpoint
        for f in _files: torch.save(ckpt_kwd, self.project / f)

    def bp_gradient(self, loss):
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optim)
        self._scaler.update()
        self._optim.zero_grad()
        return torch.isfinite(loss)

    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def metrics(self, generator) -> np.ndarray:
        for batch in generator: pass
        raise NotImplementedError

    def fitness(self, metrics) -> float:
        raise NotImplementedError

    def __call__(self, train_set, eval_set=None):
        ckpt = self.load_ckpt('last.pt')
        start_epoch = ckpt.get('epoch', 0)
        # 开始训练 / 继续训练
        result = Result(self.project, title=('loss', 'lr', *self._m_title))
        for epoch in range(start_epoch + 1, self._epochs + 1):
            dump_files = ['last.pt']
            metrics = np.array([])
            # train
            self.model.train()
            train_res = self.train(epoch, train_set)
            # eval: 由于 DropBlock 的特性, 必须执行模式切换
            self.model.eval()
            if eval_set:
                eval_res = self.eval(epoch, eval_set)
                metrics = eval_res['metrics']
                if eval_res['better']: dump_files += ['best.pt']
            # 存储 checkpoint
            self.save_ckpt(dump_files, epoch=epoch, metrics=metrics)
            # 保存结果到 Result
            result.record((train_res['avg_loss'], self._optim.param_groups[0]['lr'], *metrics), epoch)
            LOGGER.info('')
        self.finish(result)
        return self.best_fitness

    def train(self, epoch, train_set):
        LOGGER.info(fstring('epoch', 'GPU-mem', 'loss', 'lr', 'error'))
        avg_loss = tuple(self._run_one_epoch(train_set, prefix=f'{epoch}/{self._epochs}'))[0]
        self._lr_scheduler.step()
        return {'avg_loss': avg_loss}

    def eval(self, epoch, eval_set):
        with torch.no_grad():
            metrics = self.metrics(self._run_one_epoch(eval_set))
        fitness = self.fitness(metrics)
        LOGGER.info(fstring('', *metrics))
        # 保存在验证集上表现最好的网络
        better = fitness >= self.best_fitness
        if better: self.best_fitness = fitness
        return {'metrics': metrics, 'better': better}

    def finish(self, result):
        try:
            # loss, lr 归一化
            for key in ('loss', 'lr'):
                result[key] /= result[key].max()
            # 添加 fitness, 并绘制曲线图
            result['fitness'] = list(map(self.fitness, result[self._m_title].to_numpy()))
            self.plot(result, ylim=(0, 1), fitness='fitness', zerop=True, dpi=150, figsize=[12.8, 6.4])
        except Exception as reason:
            LOGGER.warning(reason)
        # 加载最优模型的参数
        model_file = self.project / 'best.pt'
        if model_file.is_file():
            ckpt = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            metrics = ckpt['metrics']
            if metrics.size > 0:
                self.best_fitness = self.fitness(metrics)
                # 输出最优模型的信息
                LOGGER.info(fstring('', *self._m_title))
                LOGGER.info(fstring('best', *ckpt['metrics']) + '\n')

    def plot(self, result, ylim=None, fitness=None, zerop=False,
             dpi=None, figsize=None, colors=None):
        if dpi: plt.rcParams['figure.dpi'] = dpi
        if figsize: plt.rcParams['figure.figsize'] = figsize
        epoch = np.array(result.index, dtype=np.int64)
        # 设置坐标系
        fig = plt.subplot()
        for key in 'right', 'top':
            fig.spines[key].set_color('None')
        # 设置坐标轴位置
        if ylim: plt.ylim(ylim)
        if zerop:
            for key in 'left', 'bottom':
                fig.spines[key].set_position(('data', 0))
            plt.xlim((0, epoch[-1]))
        # 绘制所有数据的曲线
        for key, value in result.items():
            dtype = str(value.dtype).lower()
            if ('int' in dtype or 'float' in dtype) and key != 'fitness':
                plt.plot(epoch, value, label=key, color=colors[i] if colors else None)
        # 获取所有 fitness, 并求最优轮次
        if fitness in result:
            best_epoch = result[fitness].to_numpy().argmax() + epoch[0]
            plt.xticks((epoch[0], best_epoch, epoch[-1]), (epoch[0], f'best-{best_epoch}', epoch[-1]))
            plt.plot((best_epoch,) * 2, plt.ylim(), color='gray', linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(self.project / 'curve.png')
        plt.close()

    def _run_one_epoch(self, data_set, prefix=''):
        ''' data_set: 数据集
            prefix: 进度条前缀'''
        torch.cuda.empty_cache()
        # 损失累计
        loss_sum = 0
        cuda = self.device.type != 'cpu'
        pbar = tqdm(enumerate(data_set), total=len(data_set))
        if not self.training: pbar.set_description(fstring('', *self._m_title))
        for i, batch in pbar:
            if self.training:
                with amp.autocast(enabled=cuda):
                    loss = self.loss(*batch)
                # loss 反向传播梯度, 检查损失值是否异常
                is_finite = self.bp_gradient(loss)
                loss_sum += loss.item() if is_finite else loss_sum / (i + 1e-6)
                pbar.set_description(fstring(prefix, self.cuda_memory(), loss_sum / (i + 1),
                                             self._optim.param_groups[0]['lr'], str(not is_finite)))
            else:
                yield batch
        torch.cuda.empty_cache()
        if self.training: yield loss_sum / len(data_set)


class CifarDebug(Trainer):

    def __init__(self, model, project, hyp):
        m_title = [f'Pr {i}' for i in range(10)]
        super().__init__(model, project, m_title=m_title, hyp=hyp)
        # 利用 CIFAR10 进行 debug
        import torchvision.transforms as tf
        from torchvision.datasets import CIFAR10
        datasets = [CIFAR10(f'cifar/{i}', transform=tf.ToTensor(), download=True, train=not i) for i in range(2)]
        Loader = torch.utils.data.DataLoader
        kwargs = dict(batch_size=self.hyp['batch_size'], shuffle=True)
        self.train_set = Loader(datasets[0], drop_last=True, **kwargs)
        self.eval_set = Loader(datasets[1], drop_last=False, **kwargs)
        super().__call__(self.train_set, self.eval_set)

    def loss(self, image, target):
        pred = self.model(image.to(self.device))
        return F.cross_entropy(pred, target.to(self.device))

    def metrics(self, generator):
        preds, targets = [], []
        for img, tar in generator:
            preds.append(self.model(img.to(self.device)).cpu().argmax(dim=-1))
            targets.append(tar)
        preds, targets = map(np.concatenate, (preds, targets))
        return Crosstab(preds, targets, 10).precision

    def fitness(self, metrics):
        return metrics.mean()


if __name__ == '__main__':
    a = torch.tensor([2])
    optiz = torch.optim.Adam([a], lr=0.1)
    sche = CosineLR(optiz, lrf=1e-2, epochs=100, verbose=True)
    for i in range(100): sche.step()
