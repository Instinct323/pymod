import os
import re
import sys
from pathlib import Path


def get_requirements(path=Path()):
    ''' 获取依赖包'''
    import sys
    modules = {}
    for m in list(sys.modules.values()):
        # 根据文件路径筛选
        file = m.__dict__.get('__file__', None)
        if file and 'site-packages' in file:
            # 根据导入方式筛选
            if 'SourceFileLoader' in str(m.__loader__):
                # 根据隐藏模块前缀筛选
                name = m.__name__.split('.', maxsplit=1)[0]
                if name[0] != '_': modules[name] = m
    # 筛除 site-packages 中的标准库
    for name in ['pip', 'pkg_resources', 'setuptools', 'wheel']:
        if name in modules: modules.pop(name)
    modules = list(modules)
    # 写入 txt 文本
    with open(path / 'requirements.txt', 'w') as f:
        f.writelines(map(lambda s: s + '\n', modules))
    return modules


class CondaEnv:

    def __init__(self):
        self.env = os.popen('conda info --envs')
        self.version = sys.version_info[:3]

    def create(self, name, version=None):
        version = '.'.join(version if version else self.version)
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
        for key, value in {'timeout': 6000,
                           'index-url': 'https://pypi.tuna.tsinghua.edu.cn/simple',
                           'trusted-host': 'pypi.tuna.tsinghua.edu.cn'}.items():
            os.system(f'pip config set global.{key} {value}')
        # 配置 conda
        for param in ('--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
                      '--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
                      '--set show_channel_urls yes'):
            os.system(f'conda config {param}')

    def __repr__(self):
        if not isinstance(self.env, str):
            self.env = re.search(r'\*\s+.+', self.env.read()
                                 ).group().split(maxsplit=1)[-1].rsplit('\\', maxsplit=1)[-1]
        return f'<{type(self).__name__} {self.env} {".".join(map(str, self.version))}>'


if __name__ == '__main__':
    env = CondaEnv()
    # env.jupyter(r'D:\Information\Python\Work_Space')
    env.install('pot', pip=True)
