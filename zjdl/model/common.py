import math
from typing import Optional

from timm.layers import to_2tuple, DropPath

from .fourier import FourierFeatures
from .utils import *

tuple(map(register_module("c1,c2"), (nn.Linear, nn.Conv1d, nn.Conv2d)))
tuple(map(register_module("c1"), (nn.BatchNorm2d, nn.LayerNorm)))


@register_module("c1")
class BatchNorm(nn.BatchNorm2d):

    def __init__(self, c1, s=1):
        super().__init__(c1)
        self.s = s

    def forward(self, x):
        return super().forward(x if self.s == 1 else x[..., ::self.s, ::self.s])

    def unpack(self, detach=False):
        mean, bias = self.running_mean, self.bias
        std = (self.running_var + self.eps).float().sqrt().to(mean)
        weight = self.weight / std
        eq_param = weight, bias - weight * mean
        return tuple(map(lambda x: x.data, eq_param)) if detach else eq_param


@register_module("c1,c2")
class Conv(nn.Conv2d):
    """ Conv - BN - Act"""
    deploy = property(lambda self: isinstance(self.bn, nn.Identity))

    def __init__(self, c1, c2, k=3, s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU, ctrpad=True):
        assert k & 1, "The convolution kernel size must be odd"
        # 深度可分离卷积
        if g == "dw":
            g = c1
            assert c1 == c2, "Failed to create DWConv"
        # nn.Conv2d 的关键字参数
        self.cfg_ = dict(
            in_channels=c1, out_channels=c2, kernel_size=k,
            stride=s, padding=auto_pad(k, s if ctrpad else 1, d), groups=g, dilation=d
        )
        super().__init__(**self.cfg_, bias=False)
        self.bn = BatchNorm(c2)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(super().forward(x)))

    @classmethod
    def reparam(cls, model: nn.Module):
        for m in filter(lambda m: isinstance(m, cls) and not m.deploy, model.modules()):
            kernel = m.weight.data.clone()
            bn_w, bn_b = m.bn.unpack(detach=True)
            # 合并 nn.Conv 与 BatchNorm
            m.weight.data, m.bias = kernel * bn_w.view(-1, 1, 1, 1), nn.Parameter(bn_b)
            m.bn = nn.Identity()


@register_module("c1,c2")
class RepConv(nn.Module):
    """ :param k: 卷积核尺寸, 0 表示恒等映射"""
    deploy = property(lambda self: isinstance(self.rep, nn.Conv2d))

    def __init__(self, c1, c2, k=(0, 1, 3), s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU):
        super().__init__()
        self.rep = nn.ModuleList()
        assert len(k) > 1, "RepConv with a single branch is illegal"
        for k in sorted(k):
            # Identity
            if k == 0:
                assert c1 == c2, "Failed to add the identity mapping branch"
                self.rep.append(BatchNorm(c2, s=s))
            # nn.Conv2d + BatchNorm
            elif k > 0:
                assert k & 1, f"The convolution kernel size {k} must be odd"
                self.rep.append(Conv(c1, c2, k=k, s=s, g=g, d=d, act=None, ctrpad=False))
            else:
                raise AssertionError(f"Wrong kernel size {k}")
        # Activation
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.rep(x) if self.deploy else sum_(tuple(m(x) for m in self.rep)))

    @classmethod
    def reparam(cls, model: nn.Module):
        Conv.reparam(model)
        # 查询模型的所有子模型, 对 RepConv 进行合并
        for m in filter(lambda m: isinstance(m, cls) and not m.deploy, model.modules()):
            src, cfg = m.rep[-1].weight, m.rep[-1].cfg_
            conv = nn.Conv2d(**cfg, bias=True).to(src)
            mlist, m.rep = m.rep, conv
            (c2, c1g, k, _), g = conv.weight.shape, conv.groups
            # nn.Conv2d 参数置零
            nn.init.constant_(conv.weight, 0)
            nn.init.constant_(conv.bias, 0)
            for branch in mlist:
                # BatchNorm
                if isinstance(branch, BatchNorm):
                    w, b = branch.unpack(detach=True)
                    conv.weight.data[..., k // 2, k // 2] += torch.eye(c1g).repeat(g, 1).to(src) * w[:, None]
                # Conv
                else:
                    p = (k - branch.kernel_size[0]) // 2
                    w, b = branch.weight.data, branch.bias.data
                    conv.weight.data += F.pad(w, (p,) * 4)
                conv.bias.data += b


@register_module("c1")
class DyT(nn.Module):

    def __init__(self, c1, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.w = nn.Parameter(torch.ones(c1))
        self.b = nn.Parameter(torch.zeros(c1))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.w * x + self.b


@register_module("c1,c2")
class MobileOne(nn.Sequential):

    def __init__(self, c1, c2, k=(0, 1, 3), s=1, d=1):
        super().__init__(
            RepConv(c1, c1, k=k, s=s, g="dw", d=d),
            Conv(c1, c2, 1)  # RepConv(c1, c1, k=(0, 1))
        )


@register_module("c1,c2", "n")
class ELA(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=3):
        super().__init__()
        c_ = make_divisible(c2 * e, divisor=4)
        self.ela1 = Conv(c1, c_, 1)
        self.ela2 = Conv(c1, c_, 1)
        self.elan = nn.ModuleList(
            nn.Sequential(Conv(c_, c_, 3),
                          Conv(c_, c_, 3)) for _ in range(n)
        )
        self.elap = Conv(c_ * (n + 2), c2, 1)

    def forward(self, x):
        y = [self.ela1(x), self.ela2(x)]
        for m in self.elan: y.append(m(y[-1]))
        return self.elap(torch.cat(y, 1))


@register_module("c1,c2", "n")
class CspOSA(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=4):
        super().__init__()
        c_ = make_divisible(c2 * e, divisor=4)
        n = max(2, n)
        self.osa1 = Conv(c1, c_ * 2, 1)
        self.osa2 = Conv(c1, c_ * 2, 1)
        self.osa3 = Conv(c_ * 2, c_, 3)
        self.osan = nn.ModuleList(
            Conv(c_, c_, 3) for _ in range(n - 1)
        )
        self.osap = Conv(c_ * (n + 4), c2, 1)

    def forward(self, x):
        y = [self.osa1(x), self.osa2(x)]
        y.append(self.osa3(y[-1]))
        for m in self.osan: y.append(m(y[-1]))
        return self.osap(torch.cat(y, 1))


@register_module("c1,c2", "n")
class Hourglass(nn.Module):

    def __init__(self, c1, c2, eb=.75, ec=1.125, agg="Concat", upmode="nearest", n=3):
        super().__init__()
        # 聚合函数
        agg = eval(agg) if isinstance(agg, str) else agg
        self.agg = agg()
        # 通道数分配
        c_ = make_divisible(c1 * eb, divisor=4)
        c2t = make_divisible(logspace(c2, c2 * ec ** n, n + 1), divisor=2)
        c1t = make_divisible(logspace(c_, c2t[-1], n + 1), divisor=2)
        # 核心部分的参数
        self.hgc = Conv(c1, c_, 1)
        self.t2b = nn.ModuleList()
        self.b2t = nn.ModuleList()
        self.proj = nn.ModuleList()
        for i in range(n):
            (x1, x2), (x3, x4) = c1t[i: i + 2], c2t[i: i + 2]
            self.t2b.append(Conv(x1, x2, 3, s=2))
            # Concat
            if agg is Concat:
                x_ = make_divisible(x1 * eb)
                self.b2t.append(Conv(x_ + x4, x3, 1))
                self.proj.append(Conv(x1 if i else c1, x_, 1))
            # Shortcut
            elif agg is Shortcut:
                self.b2t.append(Conv(x4, x3, 1))
                self.proj.append(Conv(x1 if i else c1, x4, 1))
            # error
            else:
                raise TypeError("Unsupported aggregate function")
        # 瓶颈部分的参数
        self.proj.append(nn.Sequential(
            Conv(c1t[-1], c1t[-1], 3),
            Conv(c1t[-1], c1t[-1], 3)
        ))
        self.upsample = partial(F.interpolate, scale_factor=2, mode=upmode)

    def forward(self, x):
        xcache = [self.hgc(x)]
        # top to bottom
        for m in self.t2b: xcache.append(m(xcache[-1]))
        xcache[0] = x
        for i, m in enumerate(self.proj): xcache[i] = m(xcache[i])
        # bottom to top
        x = xcache[-1]
        for i, m in reversed(tuple(enumerate(self.b2t))):
            x = m(self.agg([self.upsample(x), xcache[i]]))
        return x


@register_module("c1,c2")
class Bottleneck(nn.Module):

    def __init__(self, c1, c2, s=1, g=1, d=1, e=0.25):
        super().__init__()
        c_ = make_divisible(c2 * e, divisor=4)
        self.btn1 = Conv(c1, c_, 1)
        self.btn2 = Conv(c_, c_, 3, s, g, d)
        self.btn3 = Conv(c_, c2, 1, act=None)
        self.downs = nn.Identity() if c1 == c2 and s == 1 else Conv(c1, c2, 1, s, act=None)
        self.act = self.btn1.act

    def forward(self, x):
        return self.act(self.downs(x) + self.btn3(self.btn2(self.btn1(x))))


class ConvGRU(nn.Module):
    """ Convolutional GRU"""

    def __init__(self, c1, ch, k=3):
        super().__init__()
        self.convr = Conv(c1 + ch, ch, k, act=nn.Sigmoid)
        self.convz = Conv(c1 + ch, ch, k, act=nn.Sigmoid)
        self.convh = Conv(c1 + ch, ch, k, act=nn.Tanh)

    def forward(self, x, h):
        """
        :param x: input
        :param h: hidden state
        """
        xh = torch.cat((x, h), dim=1)
        z = self.convz(xh)      # update gate
        r = self.convr(xh)      # reset gate
        h_new = self.convh(torch.cat((x, r * h), dim=1))    # new hidden state
        return (1 - z) * h + z * h_new


@register_module("c1")
class ConvFFN(nn.Module):

    def __init__(self, c1, k=7, act: nn.Module = nn.ReLU):
        super().__init__()
        self.attn = nn.Sequential(
            Conv(c1, c1, k=k, g="dw"),
            nn.Conv2d(c1, c1, 1), act(),
            nn.Conv2d(c1, c1, 1)
        )

    def forward(self, x):
        return x + self.attn(x)


@register_module("c1")
class FastAttention(nn.Module):
    """
    k > 0: RepMixer-FFN
    k = 0: Self Attention-FFN
    """

    def __init__(self, c1, k=3):
        super().__init__()
        self.local = k != 0
        if self.local:
            self.fvit = RepConv(c1, c1, k=(0, k), g="dw")
        else:
            self.fvit = nn.ModuleList([
                BatchNorm(c1),
                MultiheadAttn(c1, nhead=1)
            ])
        self.ffn = ConvFFN(c1, k=7)

    def extra_repr(self):
        return f"local={self.local}"

    def forward(self, x):
        if self.local:
            x = self.fvit(x)
        else:
            x = self.fvit[0](x)
            x = x + vec2img(self.fvit[1](img2vec(x)))
        return self.ffn(x)


@register_module("c1,c2", "n")
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher

    def __init__(self, c1, c2, k=5, e=0.5, n=3):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = make_divisible(c2 * e, divisor=4)
        self.n = n
        self.spp1 = Conv(c1, c_, 1)
        self.spp2 = Conv(c_ * (n + 1), c2, 1)
        self.maxp = partial(F.max_pool2d, kernel_size=k, stride=1, padding=auto_pad(k))

    def forward(self, x):
        x = [self.spp1(x)]
        for _ in range(self.n):
            x.append(self.maxp(x[-1]))
        return self.spp2(torch.cat(x, dim=1))


class Shortcut(nn.Module):

    def forward(self, x):
        return sum_(x)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


@register_module("c1")
class SEReLU(nn.Module):
    """ Squeeze-and-Excitation Block"""

    def __init__(self, c1, r=16):
        super().__init__()
        c_ = max(4, c1 // r)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(c_, c1, 1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x * self.se(x))


class Upsample(nn.Upsample):

    def __init__(self, s=2, mode="nearest"):
        super().__init__(scale_factor=s, mode=mode)


class DropBlock(nn.Module):
    """
    :param k: size of the masking area
    :param drop: target value of drop_prob
    :cvar epochs: the number of epochs in which drop_prob reaches its target value
    :cvar scheme: drop_prob adjustment scheme
    """
    epochs = 10
    scheme = "linear"
    _progress = property(lambda self: torch.clip(self.cnt / self.epochs, min=0, max=1).item())

    @property
    def drop(self):
        # Incremental method from 0 to 1
        scale = {"const": lambda: 1,
                 "linear": lambda: self._progress,
                 }[self.scheme]()
        return self._dp_tar * scale

    def __init__(self, k=5, drop=0.1, norm=True):
        super().__init__()
        self.register_buffer("cnt", torch.tensor([0], dtype=torch.int64))
        self.k = k
        self.norm = norm
        assert self.k & 1, "The k should be odd"
        self._dp_tar = drop

    def extra_repr(self):
        return f"k={self.k}, \n" \
               f"drop={self.drop}, \n" \
               f"scheme={self.scheme}, \n" \
               f"progress={self._progress},"

    def train(self, mode=True):
        self.cnt += mode and not self.training
        super().train(mode)

    def step(self, epochs=None):
        epochs = self.epochs if not epochs else epochs
        # Check the track of drop_prob
        drop = []
        for _ in range(epochs):
            self.eval(), self.train()
            drop.append(self.drop)
        print(f"[WARNING] The drop probability has been changed to {self.drop}")
        return drop

    def forward(self, x):
        if self.training and self.drop > 0:
            # Select the center point of the masking area in the active area
            dmask = torch.bernoulli((x > 0) * (self.drop / self.k ** 2))
            kmask = 1 - (F.max_pool2d(
                dmask, kernel_size=self.k, stride=1, padding=self.k // 2
            ) if self.k > 1 else dmask)
            # Standardization in the channel dimension
            x *= kmask
            if self.norm:
                x *= np.prod(x.shape[-2:]) / kmask.sum(dim=(2, 3), keepdims=True)
        return x


@register_module("c1", "n")
class NetVLAD(nn.Module):

    def __init__(self, c1, n):
        super().__init__()
        self.conv = Conv(c1, n, 1, act=None)
        self.cluster = nn.Parameter(torch.randn(n, c1, 1, 1))

    def forward(self, x):
        w = F.softmax(self.conv(x), dim=1)
        x = (x[:, None] - self.cluster) * w[:, :, None]
        return x.sum(dim=1)


class GeM(nn.Module):
    """ Generalized-Mean Pooling"""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp_min(1e-6).pow(self.p), 1).pow(1. / self.p)


class AvgPool(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)


@register_module("c1")
class AttnPool(nn.Module):
    """ paper: Learning Transferable Visual Models From Natural Language Supervision"""

    def __init__(self, c1, p, nhead=8, drop=0., bias=True):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.normal(0, 1 / math.sqrt(c1), [p + 1, c1]))
        self.mha = MultiheadAttn(c1, nhead=nhead, drop=drop, qk_norm=False, bias=bias)

    def forward(self, x):
        x = img2vec(x)
        # x[B, L, C] —> x[B, 1+L, C]
        x = torch.cat((x.mean(dim=1, keepdims=True), x), dim=1) + self.pos_embed
        return self.mha(x[:, :1], x)[:, 0]


@register_module("c1,c2")
class MixFFN(nn.Module):
    """
    :param e: 全连接层通道膨胀比
    :param k: 深度可分离卷积的尺寸 (k<2 时不使用)
    """

    def __init__(self, c1, c2=None, e=4., k=3, drop=0.1,
                 act: Optional[nn.Module] = QuickGELU):
        super().__init__()
        c_ = max(4, round(c1 * e))
        c2 = c1 if c2 is None else c2
        self.linear1 = nn.Linear(c1, c_)
        self.dwconv = SeqConv(c_, c_, k=k, g=c_) if k > 1 else None
        self.linear2 = nn.Sequential(
            (act if act else nn.Identity)(), nn.Dropout(p=drop),
            nn.Linear(c_, c2), nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.linear1(x)
        if self.dwconv: x = self.dwconv(x)
        return self.linear2(x)


@register_module("c1,c2")
class Mlp(MixFFN):

    def __init__(self, c1, c2=None, e=4., drop=0.1,
                 act: Optional[nn.Module] = QuickGELU):
        super().__init__(c1, c2, e=e, k=0, drop=drop, act=act)


@register_module("c1")
class MixerLayer(nn.Module):

    def __init__(self, c1, p, ep=2., ec=4, drop=0.1,
                 act: Optional[nn.Module] = QuickGELU):
        super().__init__()
        self.ln1 = nn.LayerNorm(c1)
        self.mlp1 = Mlp(p, e=ep, drop=drop, act=act)
        self.ln2 = nn.LayerNorm(c1)
        self.mlp2 = Mlp(c1, e=ec, drop=drop, act=act)

    def forward(self, x):
        t = lambda x: x.transpose(1, 2)
        # x[B, L, C]
        x = x + t(self.mlp1(t(self.ln1(x))))
        return x + self.mlp2(self.ln2(x))


@register_module("c1")
class PosEmbedding(nn.Module):

    def __init__(self, c1, etype="fourier", w=None, h=None, f=10.):
        super().__init__()
        self.hw = h, w
        self.etype = etype
        if etype == "simple":
            self.embedding = nn.Parameter(torch.normal(0, 1 / math.sqrt(c1), [h * w, c1]))
        elif etype == "fourier":
            self.embedding = None
            self.weight = nn.Parameter(torch.randn(c1))
            self.ff = FourierFeatures(c1, f=f, seed=0)
        else:
            raise AssertionError(f"Unknown position embedding type <{etype}>")

    def forward(self, x):
        equal = self.hw == LOCAL.hw
        # 可学习的位置编码
        if self.etype == "simple":
            assert equal, "Simple position embeddings are scale invariant"
            y = self.embedding
        # fourier feature
        elif self.etype == "fourier":
            if self.embedding is None or not equal:
                self.hw = LOCAL.hw
                self.embedding = torch.from_numpy(self.ff(*reversed(self.hw))
                                                  ).flatten(end_dim=1).to(x)
            y = self.embedding * self.weight
        return y


@register_module("c1,c2")
class PatchEmbedding(nn.Module):
    """ Patch Embedding: image -> conv -> seq"""

    def __init__(self, c1, c2, k=14, s=14, etype="fourier", pos_kwd={}):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=auto_pad(k, s))
        self.pos_embed = PosEmbedding(c2, etype=etype, **pos_kwd)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        x = img2vec(self.proj(x))
        x = x + self.pos_embed(x)
        x = x[:, LOCAL.pmask.to(x)] if LOCAL.pmask is not None else x
        return self.norm(x)


@register_module("c1,c2")
class MaskEmbedding(nn.Module):
    """ partial patches -> all patches"""

    def __init__(self, c1, c2, etype="fourier", pos_kwd={}):
        super().__init__()
        self.proj = nn.Linear(c1, c2)
        self.mtoken = nn.Parameter(torch.randn(c2))
        self.pos_embed = PosEmbedding(c2, etype=etype, **pos_kwd)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        x = self.proj(x)
        x_new = torch.stack((self.mtoken, torch.zeros_like(self.mtoken))
                            )[LOCAL.pmask.long()][None].repeat(int(x.size(0)), 1, 1)
        for i, j in enumerate(torch.nonzero(LOCAL.pmask).flatten()):
            x_new[:, j] = x[:, i]
        return self.norm(x_new + self.pos_embed(x_new))


@register_module("c1,c2")
class SeqConv(nn.Conv2d):
    """ 序列卷积: seq -> image -> conv -> seq"""

    def __init__(self, c1, c2, k=3, s=1, g=1, d=1):
        super().__init__(c1, c2, kernel_size=k, stride=s,
                         padding=auto_pad(k, s, d), dilation=d, groups=g)

    def forward(self, x):
        return img2vec(super().forward(vec2img(x)), in_backbone=False)


@register_module("c1")
class MultiheadAttn(nn.Module):
    """
    :param nhead: 注意力头数
    :param s: SRA 的 stride (s<2 时不使用)
    :param drop: 注意力权值的 dropout
    :param bias: QKV 线性映射的偏置
    """

    def __init__(self, c1, nhead=8, s=0, drop=0.1, qk_norm=False, bias=True):
        super().__init__()
        self.nhead = nhead
        self.chead = c1 // nhead
        assert nhead * self.chead == c1, "c1 must be divisible by n"

        self.q = nn.Linear(in_features=c1, out_features=c1, bias=bias)
        self.kv = nn.Linear(in_features=c1, out_features=2 * c1, bias=bias)
        self.norm_q = nn.LayerNorm(c1) if qk_norm else nn.Identity()
        self.norm_k = nn.LayerNorm(c1) if qk_norm else nn.Identity()

        self.sr_radio = s
        if s > 1:
            self.sr = SeqConv(c1, c1, s, s)
            self.norm = nn.LayerNorm(c1)

        self.attn_drop = drop
        self.out_drop = nn.Dropout(p=drop)
        self.proj = nn.Linear(in_features=c1, out_features=c1)

    def extra_repr(self):
        return f"nhead={self.nhead},"

    def qkv_proj(self, query, key=None):
        key = query if key is None else key
        B, L, C = map(int, key.shape)
        dims = (-1, L, self.nhead, self.chead)
        # q: [B, L, C] -> [B, L, N, C_head] -> [B, N, L, C_head]
        q = self.norm_q(self.q(query))
        q = q.view(*dims).transpose(1, 2)
        # Spatial-reduction
        if self.sr_radio > 1:
            key = self.norm(self.sr(key))
        # k, v: [B, L, C] -> [B, L", N, C_head] -> [B, N, L", C_head]
        k, v = self.kv(key).chunk(2, -1)
        k = self.norm_k(k)
        k = k.view(*dims).transpose(1, 2)
        v = v.view(*dims).transpose(1, 2)
        return q, k, v

    def out_proj(self, out):
        # out[B, N, L, C_head] -> out[B, L, C]
        out = out.transpose(1, 2).flatten(start_dim=2)
        return self.out_drop(self.proj(out))

    def forward(self, query, key=None):
        q, k, v = self.qkv_proj(query, key)
        # out[B, N, L, C_head]
        # N 对浮点运算量的影响主要在 softmax
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        # attn = self.attn_drop(F.softmax(q @ k, dim=-1))
        return self.out_proj(out)


@register_module("c1")
class TranEncoder(nn.Module):

    def __init__(self, c1, e=4., nhead=8, k=0, s=0, drop=0.1, droppath=0.1):
        super().__init__()
        self.attn = MultiheadAttn(c1, s=s, nhead=nhead, drop=drop, qk_norm=True, bias=True)
        self.norm1 = nn.LayerNorm(c1)
        self.mlp = MixFFN(c1, k=k, e=e, drop=drop)
        self.norm2 = nn.LayerNorm(c1)
        self.drop_path = DropPath(droppath) if droppath > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

    @classmethod
    def repeat(cls, n, *args, **kwargs):
        return nn.Sequential(*(cls(*args, **kwargs) for _ in range(n)))


@register_module("c1,c2", "n")
class PyramidViT(nn.Module):
    """
    :param n: TranEncoder 堆叠数
    :param r: SRA 中空间缩减时的 stride (r<2 时不使用)
    :param e: TranEncoder 全连接层通道膨胀比
    :param nhead: 注意力头数
    :param drop: 注意力权值、各个层的 dropout
    :param droppath: 残差连接的 droppath
    """

    def __init__(self, c1, c2, k=3, s=2, r=4, e=4., nhead=8, drop=0.1, droppath=0., n=3):
        super().__init__()
        self.pembed = PatchEmbedding(c1, c2, k=k, s=s)
        self.encoder = TranEncoder.repeat(n, c2, nhead=nhead, k=3, s=r, e=e, drop=drop, droppath=droppath)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        return vec2img(self.norm(self.encoder(self.pembed(x))))


@register_module("c1")
class AssignAttn(MultiheadAttn):

    def __init__(self, c1, drop=0, bias=True):
        super().__init__(c1, s=0, nhead=1, drop=drop, qk_norm=False, bias=bias)

    def forward(self, query, key=None):
        q, k, v = self.qkv_proj(query, key)
        # q[B, N, Group, C_head] × k[B, N, C_head, Pixel] = attn[B, N, Group, Pixel]
        hard_sm = (lambda x, dim: F.gumbel_softmax(x, dim=dim, hard=True)) if self.training else hard_softmax
        attn = hard_sm(q @ k, dim=-2)
        # attn[B, N, Group, Pixel] × v[B, N, Pixel, C_head] = out[B, N, Group, C_head]
        out = (attn / torch.maximum(attn.detach().sum(dim=-1, keepdim=True), torch.ones(1).to(attn))) @ v
        return self.out_proj(out), attn.detach().float().cpu()


@register_module("c1", "n")
class GroupingLayer(nn.Module):
    """
    :param n: TranEncoder 深度
    :param g: Group token 的数量
    :param e: MixFFN 全连接层通道膨胀比
    :param nhead: 注意力头数
    :param drop: MHA,MixFFN 中的 dropout
    :param droppath: 残差连接的 droppath
    """
    assignment = property(lambda self: self._memory)

    def __init__(self, c1, g=8, e=4., nhead=8, drop=0.1, droppath=0., n=3):
        super().__init__()
        self.g = g
        assert g > 1, "The number of groups should be satisfied g > 1"
        self._memory = None
        encoder_cfg = dict(k=0, s=0, e=e, nhead=nhead, drop=drop, droppath=droppath)
        mlp_cfg = dict(e=e, drop=0)
        # Transformer Encoder
        self.group_token = nn.Parameter(torch.zeros(1, g, c1))
        self.encoder = TranEncoder.repeat(n, c1, **encoder_cfg)
        self.norm_x = nn.LayerNorm(c1)
        self.norm_g = nn.LayerNorm(c1)
        # Group tokens projection
        self.mlp_g = Mlp(g, **mlp_cfg)
        self.norm1 = nn.LayerNorm(c1)
        # Pre assignment
        self.pre_assign = TranEncoder(c1, **encoder_cfg)
        self.norm2 = nn.LayerNorm(c1)
        self.norm3 = nn.LayerNorm(c1)
        # Grouping Block
        self.assign = AssignAttn(c1)
        self.norm_y = nn.LayerNorm(c1)
        self.mlp_y = Mlp(c1, **mlp_cfg)

    def extra_repr(self):
        return f"group={self.g}"

    def forward(self, x):
        # Transformer Encoder
        group_token = self.group_token.repeat(int(x.size(0)), 1, 1)
        x = self.encoder(torch.cat((x, group_token), dim=1))
        x, group_token = self.norm_x(x[:, :-self.g]), self.norm_g(x[:, -self.g:])
        # Group tokens projection
        group_token = self.mlp_g(group_token.transpose(1, 2)).transpose(1, 2)
        group_token = self.norm1(group_token)
        # Pre assignment
        t = self.pre_assign
        group_token = group_token + t.drop_path(t.attn(t.norm1(group_token), self.norm2(x)))
        group_token = group_token + t.drop_path(t.mlp(t.norm2(group_token)))
        group_token = self.norm3(group_token)
        # Grouping Block
        x, self._memory = self.assign(group_token, x)
        x = x + group_token
        # input[B, HW, C] -> group[B, Group, C]
        return x + self.mlp_y(self.norm_y(x))

    @classmethod
    def segment(cls, model: nn.Module, fmap_size=None, img_size=None):
        ret = None
        for i, m in enumerate(filter(lambda m: isinstance(m, cls), model._modules.values())):
            assign = m.assignment
            ret = assign @ ret if i else assign
        if ret is not None:
            # 给定特征图尺寸
            if np.all(fmap_size):
                fmap_size = to_2tuple(fmap_size)
                ret = ret.view(*ret.shape[:2], *fmap_size)
                # 给定原图尺寸, 上采样
                if np.all(img_size):
                    img_size = to_2tuple(img_size)
                    ret = BilinearResize(ret, img_size)
            # 逐像素分类
            return ret.argmax(dim=1).data.numpy()


if __name__ == "__main__":
    a = torch.rand(2, 32, 64)

    model = MultiheadAttn(64, nhead=4)
    print(model(a).shape)
