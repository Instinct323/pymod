import multiprocessing as mp


class Memory:

    def __init__(self):
        self._manager = mp.Manager()
        self.ns = self._manager.Namespace()

    def register(self, key: str, value: str, *init_args):
        assert key != 'ns', 'Keyword \'ns\' is unavailable'
        type_ = getattr(self._manager, value, None)
        if type_:
            setattr(self, key, type_(*init_args))
        else:
            setattr(self.ns, key, eval(value)(*init_args))

    def get_ready(self):
        delattr(self, '_manager')

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def __repr__(self):
        return ', '.join(f'\'{key}\': {value}' for key, value
                         in self.state_dict().items()).join('{}')


class DataIO(Memory):
    ''' function
        send: 返回要发送的数据
        recv: 写入接收到的数据
        read: 读取接收到的数据
        write: 写入要发送的数据'''

    def __init__(self, i_len=1, o_len=1):
        super().__init__()
        self.register('i_data', 'list', [0] * i_len)
        self.register('o_data', 'list', [0] * o_len)
        self.register('i_flag', 'bool')
        self.register('o_flag', 'bool')
        self.register('ready', 'bool')
        # 删除冗余变量
        self.get_ready()

    def send(self):
        if self.ns.o_flag:
            self.ns.o_flag = False
            return tuple(self.o_data)

    def recv(self, array):
        self.i_data[:] = array
        self.ns.i_flag = True

    def read(self, latest=True):
        if self.ns.i_flag or not latest:
            self.ns.i_flag = False
            return self.i_data

    def write(self, array):
        self.o_data[:] = array
        self.ns.o_flag = True


if __name__ == '__main__':
    import time

    data = Memory()
    data.register('a', 'int', 1)
    data.register('b', 'list', [3, 3, 3])

    start = time.time()
    for i in range(10000):
        data.b[:] = [2, 3, 4]
        print(data.b)
    print((time.time() - start) / 10, 'ms')
