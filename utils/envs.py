import datetime
import os
import re
import sys


def git_push(repositories=(r'D:\Information\Python\mod',
                           r'D:\Information\Notes',
                           r'D:\Information\Notes\info',
                           r'D:\Information\Python\Library'),
             msg=f'update on {datetime.datetime.today()}'):
    for repo in repositories:
        os.chdir(repo), print(repo.center(50, '-'))
        for cmd in ('git status', 'git add .',
                    f'git commit -m "{msg}"', 'git push main master'): os.system(cmd)


class CondaEnv:

    def __init__(self, engine='pip'):
        self.env = os.popen('conda info --envs')
        self.version = sys.version_info[:3]
        self.engine = engine

    def create(self, name, version=(3, 8, 0)):
        version = '.'.join(map(str, version if version else self.version))
        os.system(f'conda create -n {name} python=={version}')

    def install(self, pack, uninstall=False, upgrade=False):
        p = [self.engine, 'uninstall -y' if uninstall else ('install' + upgrade * ' --upgrade'), '--no-cache-dir'] \
            if self.engine == 'pip' else ['conda',
                                          'uninstall -y' if uninstall else ('upgrade' if upgrade else 'install')]
        p.append(pack)
        os.system(' '.join(p))

    def load_requirements(self, file='requirements.txt'):
        os.system(f'{self.engine} install -r {str(file)} -f https://download.pytorch.org/whl/torch_stable.html')

    def clean(self):
        os.system('conda clean -ay')

    def jupyter(self, root='.', cfg=False):
        os.chdir(str(root))
        os.system('jupyter notebook' + cfg * ' --generate-config')

    def config(self):
        # 配置 pip
        for k, v in (('timeout', 6000),
                     ('index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
                     ('trusted-host', 'pypi.tuna.tsinghua.edu.cn')):
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
    os.chdir(os.getenv('dl'))

    env = CondaEnv()
    # env.jupyter(r'D:\Information\Python\Work_Space')
    # env.load_requirements(r'D:\Information\Python\mod\requirements.txt')
    # env.install('lxml')
    env.install('setuptools==59.5.0')
    git_push()
