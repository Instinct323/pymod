from math import ceil, log2, gcd


def exgcd(a, b, c):
    ''' 求解 ax + by = c
        s.t. a > b >= 0'''
    g = gcd(a, b)
    if c % g:
        return False
    # 记录是否为负数
    inv_a = a < 0
    inv_b = b < 0
    a, b = map(abs, (a, b))
    assert a >= b

    def solve(a, b):
        if b:
            # ax + by = b(y + a//b x) + (a%b)x
            ex_y, x = solve(b, a % b)
            y = ex_y - a // b * x
        else:
            # ax + 0 = c
            x, y = c // a, 0
        return x, y

    x, y = solve(a, b)
    # 按照正负进行逆变换
    x = -x if inv_a else x
    y = -y if inv_b else y
    # xa = x - b // gcd(a, b) t
    # ya = y + a // gcd(a, b) t
    x_pace = - b // g
    y_pace = a // g
    return (x, x_pace), (y, y_pace)


def prime_filter_1(n):
    ''' 质数筛选 (欧拉筛法)
        return: 质数集合'''
    # 质数标记、质数集合
    is_prime, prime_set = [True] * (n + 1), []
    # 枚举 [2, n]
    for i in range(2, n + 1):
        # 当有质数标记, 添加到质数集合
        if is_prime[i]: prime_set.append(i)
        # 标记合数
        for p in prime_set:
            comp = i * p
            if comp > n: break
            is_prime[comp] = False
            # 退出: i % p == 0
            if i % p == 0: break
    return prime_set


class Seg_Tree:
    ''' 线段树
        key: 树结点的求值函数'''

    def __init__(self, left, right):
        self.value = None
        self.l, self.r = left, right
        self._is_leaf = left == right
        # 创建左右子树
        if not self._is_leaf:
            self.mid = (left + right) // 2
            self._children = [Seg_Tree(left, self.mid),
                              Seg_Tree(self.mid + 1, right)]

    @staticmethod
    def key(args):
        ''' 线段树求值函数
            return: 与叶结点 value 的形式保持一致'''
        return max(args)

    def check(self, *args):
        ''' 检查访问是否越界'''
        left, right = args if len(args) == 2 else args * 2
        return self.l <= left and right <= self.r

    def update(self):
        ''' 更新树结点的值'''
        if not self._is_leaf:
            args = [child.update() for child in self._children]
            self.value = self.key(args)
        return self.value

    def __setitem__(self, idx, value):
        assert self.check(idx)
        # 设置叶结点
        if self._is_leaf:
            self.value = value
        else:
            # 查找叶结点
            for child in self._children:
                if child.check(idx):
                    child[idx] = value

    def __getitem__(self, range_):
        ''' range_: 区间 [l, r]'''
        left, right = range_
        assert self.check(left, right)
        # 区间相等
        if left == self.l and right == self.r:
            return self.value
        else:
            args = []
            # 在左子树中搜索
            if left <= self.mid:
                r_bound = min([right, self.mid])
                args.append(self._children[0][left, r_bound])
            # 在右子树中搜索
            if self.mid + 1 <= right:
                l_bound = max([self.mid + 1, left])
                args.append(self._children[1][l_bound, right])
            return self.key(args)

    def __repr__(self):
        if self._is_leaf:
            return str(self.value)
        return str(self._children)


class BinTreeNode:
    ''' 二叉树结点'''

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        id_ = f'{{TreeNode: val={self.val}'
        if self.left:
            id_ += f', left={self.left}'
        if self.right:
            id_ += f', right={self.right}'
        return id_ + '}'


def bin_tree(inorder, preorder=None, postorder=None):
    ''' 构造二叉树:
        inorder: LDR
        preorder: DLR
        postorder: LRD'''
    if not (preorder or postorder):
        raise AssertionError('需添加先序/后序遍历序列')
    if postorder:
        preorder = list(reversed(postorder))
        # LRD -> DRL
    pre_index = [preorder.index(val) for val in inorder]

    def create(inorder, pre_index):
        if inorder:
            index = pre_index.index(min(pre_index))
            val = inorder[index]
            left = create(inorder[:index], pre_index[:index])
            right = create(inorder[index + 1:], pre_index[index + 1:])
            return BinTreeNode(val, left, right)
        else:
            return None

    return create(inorder, pre_index)


class TreeNode:
    ''' 边权树结点'''

    def __init__(self, val):
        self.val = val
        self.children = []

    def __add__(self, other):
        node, weight = other
        assert isinstance(node, TreeNode), 'tuple[0] 应为树结点'
        assert isinstance(weight, (int, float)), 'tuple[1] 应为边权值'
        self.children.append(other)
        return self


class Trie(dict):
    ''' 前缀树'''
    end_of_word = '#'

    def __getitem__(self, item):
        return self.get(item, None)

    def insert(self, word):
        head, *tail = word
        tar_node = self[head]
        if not tar_node:
            tar_node = Trie()
            self[head] = tar_node
        if tail:
            tar_node.insert(tail)
        else:
            tar_node[self.end_of_word] = True

    def search(self, word):
        head, *tail = word
        next_ = self[head]
        if next_ is None: return False
        # 有尾部则继续搜索，无尾部则查找单词结束符
        return next_.search(tail) if tail else bool(next_[self.end_of_word])

    def startsWith(self, word):
        head, *tail = word
        next_ = self[head]
        if next_ is None: return False
        # 有尾部则继续搜索，无尾部则返回 True
        return next_.startsWith(tail) if tail else True


def comp_bin_tree(tree):
    ''' print 完全二叉树'''
    if tree:
        loc = 0
        for level in range(ceil(log2(len(tree)))):
            print(*tree[loc: loc + 2 ** level])
            loc = loc + 2 ** level
        print(*tree[loc:], '\n')


def euler_path(source, adj, search):
    ''' 欧拉路径 (遍历边各 1 次)
        source: 路径起点
        adj: 图的邻接表
        search: 出边搜索函数'''
    path, stack = [], [source]
    while stack:
        # 访问: 栈顶顶点
        visit, out = stack[-1], adj[visit]
        if out:
            visit, out = search(visit, out)
            # 入栈: 尚未满足途径点条件
            # 入列: 满足途径点条件
            (stack if out else path).append(visit)
        # 入列: 满足途径点条件
        else:
            path.append(stack.pop(-1))
    return reversed(path)
