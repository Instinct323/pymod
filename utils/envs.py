import os
import re
import sys
from pathlib import Path


def get_requirements(path=Path()):
    ''' 获取依赖包'''
    modules = {}
    for m in sys.modules.values():
        # 根据文件路径筛选
        f = m.__dict__.get('__file__', None)
        if f and 'site-packages' in f:
            # 根据导入方式筛选
            if 'SourceFileLoader' in str(m.__loader__):
                # 根据隐藏模块前缀筛选
                n = m.__name__.split('.', maxsplit=1)[0]
                if n[0] != '_': modules[n] = m
    # 写入 txt 文本
    with open(path / 'requirements.txt', 'w') as f:
        f.writelines(map(lambda s: s + '\n', modules.keys()))
    return modules


class CondaEnv:

    def __init__(self):
        self.env = os.popen('conda info --envs')
        self.version = sys.version_info[:3]

    def create(self, name, version=(3, 8, 0)):
        version = '.'.join(map(str, version if version else self.version))
        os.system(f'conda create -n {name} python=={version}')

    def install(self, pack, pip=True, uninstall=False, upgrade=False):
        engine = 'pip' if pip else 'conda'
        param = 'uninstall -y' if uninstall else ('install' + upgrade * ' --upgrade')
        os.system(f'{engine} {param} {pack}')

    def load_requirements(self, file='requirements.txt'):
        os.system(f'pip install -r {str(file)}')

    def clean(self):
        return os.popen('conda clean -y -all')

    def jupyter(self, root='.', cfg=False):
        os.chdir(str(root))
        os.system('jupyter notebook' + cfg * ' --generate-config')

    def config(self):
        # 配置 pip
        for k, v in {'timeout': 6000,
                     'index-url': 'https://pypi.tuna.tsinghua.edu.cn/simple',
                     'trusted-host': 'pypi.tuna.tsinghua.edu.cn'}.items():
            os.system(f'pip config set global.{k} {v}')
        # 配置 conda
        for p in ('--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
                      '--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
                      '--set show_channel_urls yes'):
            os.system(f'conda config {p}')

    def __repr__(self):
        if not isinstance(self.env, str):
            self.env = re.search(r'\*\s+.+', self.env.read()
                                 ).group().split(maxsplit=1)[-1].rsplit('\\', maxsplit=1)[-1]
        return f'<{type(self).__name__} {self.env} {".".join(map(str, self.version))}>'


if __name__ == '__main__':
    env = CondaEnv()
    # env.jupyter(r'D:\Information\Python\Work_Space')
    # env.install('pot', pip=True)
