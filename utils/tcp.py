import logging
import multiprocessing as mp
import pickle
import socket
import time

import numpy as np

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def pickle_size(obj):
    return len(pickle.dumps(obj))


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


class TCP_Socket(socket.socket):
    ''' param
            port: 作为服务端时所开放的端口
            timeout: 传输数据时的超时时间

        attribute
            host: 本机网络的 IPv4 地址
            sockname: 连接成功 -> 自身地址
            peername: 连接成功 -> 对方地址'''

    def __init__(self, port=22, timeout=1e-3):
        super().__init__(family=socket.AF_INET, type=socket.SOCK_STREAM)
        tmp = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        tmp.connect(('8.8.8.8', 80))
        self.host = tmp.getsockname()[0]
        # conn: TCP 连接对象
        self._port, self._conn = port, tmp.close()
        self._timeout = timeout
        del tmp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self != self._conn and self._conn: self._conn.close()
        self.close()

    def _user(self, addr):
        return f'{addr[0]}:{addr[1]}'

    def _bind(self):
        while self._port <= 65535:
            try:
                self.bind((self.host, self._port))
                break
            except OSError as reason:
                # 如果端口号不可用
                code = str(reason).split()[1][:-1]
                if code != '10013': raise reason
                self._port += 1

    def _reduce(self):
        self._conn.settimeout(self._timeout)
        # 获取连接双方用户名
        self.sockname = self._user(self._conn.getsockname())
        self.peername = self._user(self._conn.getpeername())

    def upload(self, data):
        return self._conn.sendall(data)

    def download(self):
        return self._conn.recv(1024)

    def connect_auto(self, addr=None, timeout=30):
        self.settimeout(timeout)
        # 客户端: 向服务端请求连接
        if addr:
            peername = self._user(addr)
            try:
                LOGGER.info(f'Trying to connect server {peername}')
                self.connect(addr)
                # 记录 TCP 连接对象
                self._conn = self
                self._reduce()
                # 输出连接日志
                LOGGER.info(f'User {self.sockname} is logged in')
                LOGGER.info(f'Connected to server {self.peername}')
                return True
            except (socket.timeout, ConnectionRefusedError):
                LOGGER.warning(f'User {peername} is not online')
        # 服务端: 等待客户端的连接请求
        else:
            self._bind()
            self.listen(1)
            LOGGER.info(f'Server {self._user(self.getsockname())} is logged in')
            try:
                # 记录 TCP 连接对象
                self._conn, addr = self.accept()
                self._reduce()
                # 输出连接日志
                LOGGER.info(f'Connected to user {self.peername}')
                return True
            except socket.timeout:
                LOGGER.warning('No connection requests were listened for')


def chat(input_func, addr=None, timeout=1e-3, exit_code=r'\exit', encoding='utf-8'):
    with TCP_Socket(timeout=timeout) as tcp:
        try:
            if tcp.connect_auto(addr=addr):
                LOGGER.info(f'Enter the chat room')
                while True:
                    send = input_func()
                    # 发送缓冲区的数据
                    if send:
                        try:
                            tcp.upload(send.encode(encoding))
                            print(f'{tcp.sockname} << {send}')
                        except socket.timeout:
                            print(f'Fail to send << {send}')
                        if send == exit_code: break
                    # 接收对方发送的数据
                    try:
                        recv = tcp.download().decode(encoding)
                        if recv:
                            if recv == exit_code: break
                            print(f'{tcp.peername} << {recv}')
                    except socket.timeout:
                        pass
                LOGGER.info(f'Exit the chat room')
        except Exception as reason:
            LOGGER.error(f'{type(reason).__name__}: {reason}')


def transfer_async(dataio, addr=None, timeout=1e-4):
    ''' dataio: DataIO 实例
        addr: 服务端地址
        timeout: 传输数据时的超时时间'''
    t_recv, t_send, t_wait, momentum = (0,) * 3 + (0.1,)
    with TCP_Socket(timeout=timeout) as tcp:
        try:
            if tcp.connect_auto(addr=addr):
                LOGGER.info(f'The data transmission channel is enabled')
                dataio.ns.ready = True
                while True:
                    t0 = time.time()
                    # 发送数据
                    send = dataio.send()
                    if send is not None:
                        try:
                            tcp.upload(pickle.dumps(send))
                            t_send = momentum * (time.time() - t0) * 1e3 + (1 - momentum) * t_send
                            t0 = time.time()
                        except socket.timeout:
                            pass
                        except pickle.PickleError as reason:
                            LOGGER.warning(f'{type(reason).__name__}: {reason}')
                    # 接收对方发送的数据
                    try:
                        recv = pickle.loads(tcp.download())
                        dataio.recv(recv)
                        t_recv = momentum * (time.time() - t0) * 1e3 + (1 - momentum) * t_recv
                        t0 = time.time()
                    except socket.timeout:
                        pass
                    except pickle.PickleError as reason:
                        LOGGER.warning(f'{type(reason).__name__}: {reason}')
                    # 输出网络迟延
                    t_wait = momentum * (time.time() - t0) * 1e3 + (1 - momentum) * t_wait
                    print('\r' + (' ' * 4).join(map(lambda s, t: f'T-{s}: {t:.2f} ms',
                                                    ('recv', 'send', 'wait'), (t_recv, t_send, t_wait))), end='')
                LOGGER.info(f'End of data transmission')
        except Exception as reason:
            LOGGER.error(f'{type(reason).__name__}: {reason}')


if __name__ == '__main__':
    chat_flag = False

    if chat_flag:
        import os
        from pathlib import Path

        INPUT_FILE = Path(os.getenv('lab')) / 'data/main.txt'


        def input_func():
            data = INPUT_FILE.read_text(encoding='utf-8')
            INPUT_FILE.write_text('')
            return data


        chat(input_func)

    else:
        from mod.zjplot import rainbow
        from tqdm import tqdm

        import matplotlib.pyplot as plt

        dataio = DataIO(3, 3)
        t = round(1e4)

        tcp_com = mp.Process(target=transfer_async, args=(dataio,))
        tcp_com.start()

        # 等待 TCP 连接成功
        while not dataio.ns.ready: pass

        for color in tqdm(rainbow[1: 5]):
            y = [0]
            t0 = time.time()

            # 发送状态量, 接收控制量
            for i in range(t):
                dataio.write((y[-1],) * 3)
                y.append(dataio.i_data[0])

            # 得到运行时间
            cost = time.time() - t0
            dataio.write((0,))
            time.sleep(1)

            rate = round(cost * 1e3 / y[-1], 2)
            x = np.linspace(0, cost, t + 1)
            plt.plot(x, y, color=color, label=f'{rate} ms')

        plt.legend()
        plt.show()
