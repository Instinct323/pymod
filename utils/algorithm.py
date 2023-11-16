import heapq
import itertools as it
import math
import operator
import random


class DisjointSet:
    ''' 并查集'''

    def __init__(self, length):
        # 记录前驱结点, 结点级别
        self._pre = list(range(length))
        self._rank = [1] * length

    def find(self, i):
        while self._pre[i] != i:
            i = self._pre[i]
        return i

    def is_same(self, i, j):
        return self.find(i) == self.find(j)

    def join(self, i, j):
        i, j = map(self.find, [i, j])
        # 前驱不同, 需要合并
        if i != j:
            # 访问前驱级别
            rank_i, rank_j = self._rank[i], self._rank[j]
            # 前驱级别相同: 提高一个前驱的级别, 作为根结点
            if rank_i == rank_j:
                self._rank[i] += 1
                self._pre[j] = i
            # 前驱级别不同: 级别高的作为根结点
            else:
                self._pre[j] = i if rank_i > rank_j else j

    def __repr__(self):
        return str(self._pre)


class SegmentTree:
    ''' :param oper: 二元运算函数
        :param v0: 满足 oper(x, v0) = x 的值

        :ivar index: 数组元素在二叉树的结点索引
        :ivar tree: 以二叉树形式存储的聚合值'''

    def __init__(self, arr, oper=operator.add, v0=0):
        self.arr = arr
        self.oper = oper
        self.v0 = v0
        self.index = [-1] * len(self)
        # 从根结点开始建树
        self.tree = 4 * len(self) * [self.v0]
        self.build(0, 0, len(self) - 1)
        # 计算存储树所需要的数组长度
        self.tree = self.tree[:max(self.index) + 1]

    def __len__(self):
        return len(self.arr)

    def _slice_handler(self, i: slice, func, *args, **kwargs):
        self._l, self._r = i.start, i.stop - 1
        ret = func(*args, **kwargs)
        for k in ('_l', '_r'): delattr(self, k)
        return ret

    def __setitem__(self, i, v):
        if isinstance(i, int):
            # 修改数组值
            node = self.index[i]
            self.arr[i] = self.tree[node] = v
        raise NotImplementedError

    def __getitem__(self, i):
        return self.arr[i] if isinstance(i, int) else \
            self._slice_handler(i, self.query, 0, 0, len(self) - 1)

    def get_child(self, node, s, e):
        mid = (s + e) // 2
        # 在二叉树中的结点索引, 区间起点, 区间终点
        return (
            (2 * node + 1, 2 * node + 2),
            (s, mid + 1), (mid, e)
        )

    def build(self, node, s, e):
        # __init__ 隐式调用
        if s == e:
            # 存储叶结点索引
            self.index[s] = node
            self.tree[node] = self.arr[s]
        else:
            # 聚合左右孩子结点
            self.tree[node] = self.oper(
                *map(self.build, *self.get_child(node, s, e))
            )
        return self.tree[node]

    def query(self, node, s, e):
        # __getitem__ 隐式调用
        # 当前区间 \subseteq 查询区间
        if self._l <= s and e <= self._r: return self.tree[node]
        # 当前区间 \cap 查询区间 = \emptyset
        if e < self._l or self._r < s: return self.v0
        # 当前区间 \cap 查询区间 \neq \emptyset
        return self.oper(
            *map(self.query, *self.get_child(node, s, e))
        )


class PdrChecker:
    ''' hash 算法判断回文串
        :param hashv: 字符串的 hash 值对应的列表
        :param base: max(hashv) + 1

        :example:
        >>> hashv = [ord(c) - 97 for c in "racecar"]
        >>> checker = PdrChecker(hashv, 26)
        >>> checker[1:6]
        True'''

    def __init__(self, hashv, base):
        self.base = base
        # 正/逆序 hash 序列
        self.hseq = self.get_seq(hashv)
        self.hseq_r = self.get_seq(hashv[::-1])

    def get_seq(self, hashv):
        seq = [0] * (len(hashv) + 1)
        for i in range(len(hashv)):
            seq[i + 1] = seq[i] * self.base + hashv[i]
        return seq

    def __getitem__(self, item: slice):
        s, e = item.start, item.stop
        gain = self.base ** (e - s)
        return self.hseq[e] - self.hseq[s] * gain == self.hseq_r[- s - 1] - self.hseq_r[- e - 1] * gain


def rem_theorem(mods, rems, lcm_fcn=math.prod):
    ''' 中国剩余定理
        :param mods: 模数集
        :param rems: 余数集
        :param lcm_fcn: 最小公倍数的求解函数 (模数集全为质数时使用 math.prod)
        :return: 满足给定条件的余数项'''
    lcm = lcm_fcn(mods)
    # 费马小定理求逆元, 要求 a,p 互质
    inv = lambda a, p: pow(a, p - 2, p)
    result = 0
    for p, r in zip(mods, rems):
        a = lcm // p
        result += r * a * inv(a, p)
    return result % lcm


def prime_filter(n):
    ''' 质数筛选 (埃氏筛法)
        :return: 质数标志

        :example:
        >>> is_prime = prime_filter(10000)
        >>> is_prime[2:].count(True)
        1229'''
    is_prime = [True] * (n + 1)
    # 枚举 [2, sqrt(n)]
    for i in range(2, math.isqrt(n) + 1):
        if is_prime[i]:
            for c in range(i ** 2, n + 1, i):
                is_prime[c] = False
    return is_prime


def map_euler_fun(n):
    ''' 批算欧拉函数'''
    # 素数集合、素数标记、欧拉函数值
    prime_set = []
    is_prime = [True] * (n + 1)
    value = [0] * (n + 1)
    # 枚举 [2, n]
    for i in range(2, n + 1):
        # 当有素数标记, 添加到素数集合
        if is_prime[i]:
            prime_set.append(i)
            value[i] = i - 1
        # 标记合数
        for p in prime_set:
            # 合数 = i × 素数
            c = i * p
            # 退出: 合数越界
            if c > n: break
            # 标记: 合数
            is_prime[c] = False
            if i % p == 0:
                # i 是 p 的倍数
                value[c] = value[i] * p
                break
            else:
                # i 和 p 互质: 利用积性计算
                value[c] = value[i] * (p - 1)
    return value


def try_div(n, factor={}):
    ''' 试除法分解'''
    i, bound = 2, math.isqrt(n)
    while i <= bound:
        if n % i == 0:
            # 计数 + 整除
            cnt = 1
            n //= i
            while n % i == 0:
                cnt += 1
                n //= i
            # 记录幂次, 更新边界
            factor[i] = factor.get(i, 0) + cnt
            bound = math.isqrt(n)
        i += 1
    if n > 1: factor[n] = 1
    return factor


def all_factor(n):
    ''' 所有因数'''
    prime = try_div(n)
    factor = [1]
    for i in prime:
        tmp = []
        for p in map(lambda x: i ** x, range(1, prime[i] + 1)):
            tmp += [p * j for j in factor]
        factor += tmp
    return factor


def miller_rabin(p):
    ''' 素性测试'''
    # 特判 4
    if p <= 4: return p in (2, 3)
    # 对 p-1 进行分解
    pow_2, tmp = 0, p - 1
    while tmp % 2 == 0:
        tmp //= 2
        pow_2 += 1
    # 进行多次素性测试
    for a in (2, 3, 5, 7, 11):
        basic = pow(a, tmp, p)
        # a^m 是 p 的倍数或者满足条件
        if basic in (0, 1, p - 1): continue
        # 进行 r-1 次平方
        for _ in range(1, pow_2):
            basic = basic ** 2 % p
            # 怎样平方都是 1
            if basic == 1: return False
            # 通过 a 的素性测试
            if basic == p - 1: break
        # 未通过 a 的素性测试
        if basic != p - 1: return False
    # 通过所有 a 的素性测试
    return True


def pollard_rho(n):
    ''' 求因数: 7e5 以上'''
    # 更新函数
    bias = random.randint(3, n - 1)
    update = lambda i: (i ** 2 + bias) % n
    # 初始值
    x = random.randint(0, n - 1)
    y = update(x)
    # 查找序列环
    while x != y:
        factor = math.gcd(abs(x - y), n)
        # gcd(|x - y|, n) 不为 1 时, 即为答案
        if factor != 1: return factor
        x = update(x)
        y = update(update(y))
    return n


class prime_factor(dict):
    ''' 质因数分解
        require: miller_rabin, pollard_rho'''

    def __init__(self, n):
        super().__init__()
        self.main(n, gain=1)

    def add(self, n, cnt):
        self[n] = self.get(n, 0) + cnt

    def count(self, n, fac):
        # 试除并记录幂次
        cnt = 1
        n //= fac
        while n % fac == 0:
            cnt += 1
            n //= fac
        return n, cnt

    def main(self, n, gain):
        # 试除法求解
        if n < 7e5: return self.try_divide(n, gain=gain)
        # 米勒罗宾判素
        if miller_rabin(n): return self.add(n, gain)
        # pollard rho 求解因数
        fac = pollard_rho(n)
        n, cnt = self.count(n, fac)
        # 递归求解因数的因数
        self.main(fac, gain=cnt * gain)
        # 递归求解剩余部分
        if n > 1: self.main(n, gain=gain)

    def try_divide(self, n, gain=1):
        ''' 试除法分解'''
        i, bound = 2, math.isqrt(n)
        while i <= bound:
            if n % i == 0:
                # 计数 + 整除
                n, cnt = self.count(n, i)
                # 记录幂次, 更新边界
                self.add(i, cnt * gain)
                bound = math.isqrt(n)
            i += 1
        if n > 1: self.add(n, gain)


def next_perm(seq):
    ''' 找到下个字典序
        e.g.: 8 3 7 6 5 4 2 1
                |       |    '''
    n = len(seq)
    filt1 = lambda i: seq[i] >= seq[i + 1]
    try:
        l = next(it.dropwhile(filt1, range(n - 2, -1, -1)))
        # 找到交换位
        filt2 = lambda r: seq[l] >= seq[r]
        r = next(it.dropwhile(filt2, range(n - 1, l, -1)))
        seq[l], seq[r] = seq[r], seq[l]
        # 逆转逆序区
        seq[l + 1:] = reversed(seq[l + 1:])
        return seq
    except StopIteration:
        return None


def manacher(string):
    ''' 最长回文串长度'''
    string = '#'.join(string.join('^$'))
    center, border = 0, 0
    # 以对应字符为中心 最长回文串的半径
    p = [0] * len(string)
    is_same = lambda i0, i1: i0 >= 0 and i1 < len(string) and string[i0] == string[i1]
    for i in range(1, len(string) - 1):
        # 利用回文串的对称性进行赋值
        p[i] = min(max(0, border - i), p[max(0, 2 * center - i)])
        # 中心扩展法
        while is_same(i - p[i] - 1, i + p[i] + 1): p[i] += 1
        # 更新回文串中心, 回文串右端点 ('#')
        if i + p[i] > border: center, border = i, i + p[i]
    return max(p)


def dijkstra(source, adj):
    ''' 单源最短路径 (不带负权)
        :param source: 源点
        :param adj: 图的邻接表'''
    n = len(adj)
    # 记录单源最短路, 未访问标记
    info = [[float('inf'), True] for _ in range(n)]
    info[source][0] = 0
    # 记录未完成搜索的点 (优先队列)
    undone = [(0, source)]
    while undone:
        # 找到离源点最近的点作为中间点 m
        m = heapq.heappop(undone)[1]
        if info[m][1]:
            info[m][1] = False
            # 更新单源最短路
            for i in filter(lambda j: info[j][1], adj[m]):
                tmp = info[m][0] + adj[m][i]
                if info[i][0] > tmp:
                    info[i][0] = tmp
                    heapq.heappush(undone, (tmp, i))
    return info


def spfa(source, adj):
    ''' 单源最短路径 (带负权)
        :param source: 源点
        :param adj: 图的邻接表'''
    n, undone = len(adj), [(0, source)]
    # 单源最短路, 是否在队, 入队次数
    info = [[float('inf'), False, 0] for _ in range(n)]
    info[source][0] = 0
    while undone:
        # 队列: 弹出中间点
        m = heapq.heappop(undone)[1]
        info[m][1] = False
        # 更新单源最短路
        for i in adj[m]:
            tmp = info[m][0] + adj[m][i]
            if info[i][0] > tmp:
                cnt = info[i][-1]
                # 入队: 被更新点
                if not info[i][1]:
                    cnt += 1
                    heapq.heappush(undone, (tmp, i))
                    # 终止: 存在负环
                    if cnt > n: return False
                info[i] = [tmp, True, cnt]
    return info


def floyd(adj):
    ''' 多源最短路径 (带负权)
        :param adj: 图的邻接矩阵'''
    # import itertools as it
    n = len(adj)
    for m in range(n):
        for i, j in it.combinations(it.chain(range(m), range(m + 1, n)), 2):
            adj[i][j] = min(adj[i][j], adj[i][m] + adj[m][j])


def topo_sort(in_degree, adj):
    ''' AOV 网拓扑排序 (最小字典序)
        :param in_degree: 入度表
        :param adj: 图的邻接表'''
    undone = [i for i, v in enumerate(in_degree) if v == 0]
    heapq.heapify(undone)
    order = []
    while undone:
        v = heapq.heappop(undone)
        order.append(v)
        # 删除该结点, 更新入度表
        for i in adj[v]:
            in_degree[i] -= 1
            if in_degree[i] == 0: heapq.heappush(undone, i)
    return order if len(order) == len(in_degree) else False


def prim(source, adj):
    ''' 最小生成树
        :param source: 源点
        :param adj: 图的邻接表'''
    edges, n = [], len(adj)
    # 未完成搜索的结点
    undone = [(w, i) for i, w in adj[source].items()]
    heapq.heapify(undone)
    # 和树的最小距离, 最近结点, 未完成标志
    info = [[adj[source].get(i, float('inf')), source, True] for i in range(n)]
    info[source][-1] = False
    while undone:
        # 未被选取的顶点中, 离树最近的点
        v = heapq.heappop(undone)[1]
        if info[v][-1]:
            info[v][-1] = False
            edges.append((info[v][1], v))
            # 更新最近结点
            for i in adj[v]:
                if info[i][0] > adj[v][i]:
                    info[i][:2] = adj[v][i], v
                    heapq.heappush(undone, (adj[v][i], i))
    return edges


def wythoff_game(x):
    ''' 威佐夫博弈'''
    coef = (5 ** 0.5 + 1) / 2
    # 两塔同取时的决策 (索引为高度差)
    k, singular = 0, []
    while True:
        ak = int(coef * k)
        bk = ak + k
        if bk > x: break
        k += 1
        singular.append((ak, bk))
    # 只取一塔时的决策
    one = [float('inf')] * (x + 1)
    for ak, bk in singular:
        one[ak], one[bk] = bk, ak
    return list(enumerate(one)), singular


if __name__ == '__main__':
    one, singular = wythoff_game(100)
    print(*one, sep='\n')
    print(len(singular))
