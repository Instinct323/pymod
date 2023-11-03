import copy
import time
from pathlib import Path
from typing import Union

import thop
import torch.onnx
import yaml

from .common import *


def abs_idx(idx: Union[int, list], n: int):
    return [i % n for i in ([idx] if isinstance(idx, int) else idx)]


def switch_branch(state_dict, branch='ema'):
    branch += '.'
    # Switches to the weight of the specified branch
    for k in tuple(state_dict.keys()):
        if k.startswith(branch):
            state_dict[k[len(branch):]] = state_dict.pop(k)
    return state_dict


def cfg_modify(yaml_cfg: dict, modify: Union[tuple, list]):
    yaml_cfg = copy.deepcopy(yaml_cfg)
    # 使用特定语法修改配置
    for i, key, value in modify:
        # number, module, args
        j = 'nma'.index(key[0]) + 1
        if j < 3:
            yaml_cfg['architecture'][i][j] = value
        else:
            yaml_cfg['architecture'][i][j][int(key[1:])] = value
    return yaml_cfg


class YamlModel(nn.Module):
    ''' :param yaml_cfg:
            depth_multiple: 模块深度增益
            width_multiple: 卷积宽度增益
            fixed_layers: 不受增益影响的层索引

            in_channals: 输入的通道数量
            img_size: 输入的图像尺寸

            freeze: 欲冻结的层切片
            architecture:
                from: 输入来源
                number: 串联深度 / 模块 n 参数
                module: 模块名称
                args: 模块初始化参数 (c2, ...)'''
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self, yaml_cfg: Union[Path, dict], ch_divisor: int = 4):
        super().__init__()
        self.cfg = yaml_cfg if isinstance(yaml_cfg, dict) \
            else yaml.load(yaml_cfg.read_text(), Loader=yaml.Loader)
        # 缺省参数: 深度增益, 宽度增益
        self.cfg.setdefault('depth_multiple', 1.)
        self.cfg.setdefault('width_multiple', 1.)
        # 输入张量信息
        self.cfg.setdefault('in_channels', 3)
        img_size = self.cfg.setdefault('img_size', None)
        if img_size:
            self.cfg['img_size'] = to_2tuple(img_size)
        # 模型架构信息, 参数固定层信息
        assert self.cfg.get('architecture', None), '\"architecture\" is not defined'
        self.cfg['fixed_layers'] = [i % len(self.cfg['architecture']) for i in self.cfg.get('fixed_layers', [])]
        # 解析架构信息
        self.main = nn.ModuleList(self.parse_architecture(ch_divisor))
        # 冻结层信息
        self.cfg.setdefault('freeze', [])
        assert isinstance(self.cfg['freeze'], list), '\"freeze\" should be a list of slicing expressions'
        i = list(range(len(self.main)))
        for slc in self.cfg['freeze']:
            for i in eval(f'i[{slc}]'): self.freeze(i)
        self.init_param(), self.eval()

    def init_param(self):
        for m in self.modules():
            # fan_in: 输入结点数
            # fan_out: 输出结点数
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # BatchNorm
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Activation
            elif isinstance(m, (nn.ReLU, nn.LeakyReLU)):
                m.inplace = False

    def example_input(self, b=1):
        return torch.rand([b, self.cfg['in_channels'], *self.cfg['img_size']])

    def forward_feature(self, x, tarlayer=-1, profile=False):
        output = []
        for i, m in enumerate(self.main[:tarlayer % len(self.main) + 1]):
            if m.f != -1: x = output[m.f] if isinstance(m.f, int) else [output[f] for f in m.f]
            if profile: yield m, x
            # forward propagation
            x = m(x)
            output.append(x)
        if not profile: yield output

    def forward(self, x, tarlayer=-1):
        # imgae: uint8 -> float
        if x.dtype == torch.uint8: x = x.float() / 255
        # 获取所有层的输出并筛选
        tarlayer = abs_idx(tarlayer, n=len(self.main))
        output = next(self.forward_feature(x, max(tarlayer), profile=False))
        return output[-1] if len(tarlayer) == 1 else [output[i] for i in tarlayer]

    def profile(self, x=None, repeat=5):
        x = self.example_input() if x is None else x
        information = np.zeros(3)
        print(f"\n    {'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  module")
        for m, x in self.forward_feature(x, profile=True):
            # 测试该模块的性能
            t0 = time.time()
            for _ in range(repeat): m(x)
            cost = (time.time() - t0) / repeat
            # GFLOPs
            flops = thop.profile(m, (x,), verbose=False)[0] / 1e9
            print(f'{m.i:>3} {cost * 1e3:10.2f} {flops / max(1e-9, cost):10.2f} {m.np:10.0f}  {m.t}')
            # 完成性能测试
            information += cost, flops, m.np
        # 输出模型的性能测试结果
        cost, flops, params = information
        print(f'    {cost * 1e3:10.2f} {flops / cost:10.2f} {int(params):10}  Total')
        return information

    def pop(self, index):
        self.main._modules.pop(str(index))

    def freeze(self, index):
        print(f'freezing layer {index} <{self.main[index].t}>')
        for k, v in self.main[index].named_parameters():
            v.requires_grad = False

    def unfreeze(self, index):
        print(f'unfreezing layer {index} <{self.main[index].t}>')
        for k, v in self.main[index].named_parameters():
            v.requires_grad = True

    def simplify(self, inplace=True):
        model = (self if inplace else copy.deepcopy(self)).eval()
        for f in (Conv.reparam, RepConv.reparam): f(model)
        return model

    def onnx(self, file, branch=None, x=None):
        x = self.example_input().to(self.device) if x is None else x
        m = getattr(self, branch) if branch else self
        torch.onnx.export(m, (x,), file, opset_version=11)

    def torchscript(self, x=None):
        x = self.example_input() if x is None else x
        model = torch.jit.trace(self, (x,))

        def profile(x, repeat=5):
            t0 = time.time()
            for _ in range(repeat): model(x)
            return (time.time() - t0) / repeat * 1e3

        model.profile = profile
        print('The runtime can be obtained using TorchScript\'s function <profile>')
        return model

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            try:
                return super().load_state_dict(state_dict, strict=True)
            except Exception as error:
                LOGGER.warning(f'[WARNING] {error}')
                print('[INFO] Try loading the state_dict loosely')
        redundantp = []
        # 处理 shape 失配的参数
        local_sdict = self.state_dict()
        for key in copy.deepcopy(state_dict):
            if key in local_sdict:
                pl, pn = local_sdict[key], state_dict[key]
                sl, sn = pl.shape, pn.shape
                # 跳过标量, 不处理
                if sl != sn:
                    sl, sn = map(torch.tensor, (sl, sn))
                    s1, *s2 = torch.minimum(sl[:2], sn[:2])
                    # 对于 1D, 2D 参数, 直接裁剪
                    if pl.ndim == 1:
                        pl[:s1] = pn[:s1]
                    else:
                        s2 = s2[0]
                        if pl.ndim == 2:
                            pl[:s1, :s2] = pn[:s1, :s2]
                        # 对于 4D 参数, 默认为卷积核
                        elif pl.ndim == 4:
                            k = min(sl[-1], sn[-1])
                            pad = max(0, (sl[-1] - k) // 2)
                            cut = max(0, (sn[-1] - k) // 2)
                            pl[:s1, :s2, pad:pad + k, pad:pad + k] = pn[:s1, :s2, cut:cut + k, cut:cut + k]
                        else:
                            raise AssertionError(f'Unknown parameter: {key}')
                    # 覆盖写入现模型的参数字典
                    state_dict.pop(key)
            else:
                state_dict.pop(key)
                redundantp.append(key)
        if redundantp: LOGGER.warning(f'Redundant parameter: {", ".join(redundantp)}')
        super().load_state_dict(state_dict, strict=False)
        return self

    def parse_architecture(self, ch_divisor=4):
        print('\n%3s %17s %2s %9s  %-15s %-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
        modules, channels = [], [self.cfg['in_channels']]
        for i, (from_, number, module, args) in enumerate(copy.deepcopy(self.cfg['architecture'])):
            kwargs = {}
            # 使用深度增益对 number 进行修改
            if i not in self.cfg['fixed_layers']: number = max(1, int(number * self.cfg['depth_multiple']))
            # 对 module 进行转换
            module = eval(module) if isinstance(module, str) else module
            # 对 args 进行转换
            for j, arg in enumerate(args):
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg
                except (NameError, AttributeError):
                    pass
            # 增添 n 参数
            if module in module_required['n']: kwargs['n'], number = number, 1
            # 根据模型的参数要求, 对 args 进行修改
            if isinstance(from_, int):
                # 需要 c1,c2 参数的模块: 改变维度
                if module in module_required['c1,c2']:
                    if i not in self.cfg['fixed_layers']:
                        # 将 c2 处理为 4 的倍数
                        args[0] = make_divisible(args[0] * self.cfg['width_multiple'], divisor=ch_divisor)
                    c1, c2 = channels[from_], args[0]
                    # 升维单元、降维单元不可堆叠
                    if c1 != c2 and number != 1:
                        number = 1
                        warnings.warn('The dimension transform convolution is not stackable')
                    args.insert(0, c1)
                # 无需 c2 则视为不改变通道数
                else:
                    c2 = channels[from_]
                    # 要求 c1 参数的模块: BatchNorm2d, CBAM
                    if module in module_required['c1']: args.insert(0, c2)
                    # 除此之外: 池化层, 激活函数, DropBlock
            else:
                c1 = [channels[f] for f in from_]
                if module is Concat: c2 = sum(c1)
            # 实例化模块
            type_ = module.__name__
            module = module(*args, **kwargs) if number == 1 else \
                nn.Sequential(*[module(*args, **kwargs) for _ in range(number)])
            # 记录输出通道数
            if i == 0: channels = []
            channels.append(c2)
            # 计算网络层的参数量, 设置网络层属性
            params = sum([p.numel() for p in module.parameters()])
            module.i, module.f, module.np, module.t = i, from_, params, type_
            modules.append(module)
            # 保存除 -1 之外的 from 参数到 save 列表 (同时对负数求余)
            # if from_ != -1: self.save |= {f % i for f in ([from_] if isinstance(from_, int) else from_)}
            # 输出模块信息
            if kwargs: args.append(kwargs)
            print('%3s %17s %2s %9.0f  %-15s %-30s' % (i, from_, number, params, type_, args))
        # 输出模型的统计信息
        params = sum([m.np for m in modules])
        mbytes = params * 4 / 1e6
        print(f'\nModel Summary: {len(modules)} layers, {params} parameters, {mbytes:.3f} MB')
        return modules


if __name__ == '__main__':
    import os

    os.chdir(r'config')
    image = torch.rand([1, 3, 224, 224]).cuda().half()

    model1 = YamlModel(Path('ResNet-50.yaml')).eval().cuda().half()
    # OnnxModel.test(model1, (image,), 'model.onnx')
    # model1.profile(image)

    '''traced = model1.torchscript(image.repeat(8, 1, 1, 1)).cuda()
    traced.save(Path('ResNet-50.pt'))'''
