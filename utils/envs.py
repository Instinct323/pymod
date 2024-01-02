import datetime
import os
from pathlib import Path

execute = lambda x: print(x) or os.system(x)

if os.name == 'nt':
    # Window
    ENV = Path('D:/Information/Data/envs/cv')
    pip = ENV / 'Scripts/pip'
    jupyter = ENV / 'Scripts/jupyte'
    conda = Path('D:/Software/Anaconda3/condabin/conda')
else:
    # Linux
    ENV = Path('/home/slam602/.conda/envs/torch')
    pip = ENV / 'bin/pip'
    conda = Path('/opt/miniconda/bin/conda')


def git_push(*repositories,
             msg=f'update on {datetime.datetime.today()}'):
    for repo in repositories:
        os.chdir(repo), print(repo.center(50, '-'))
        for cmd in ('git status', 'git add .',
                    f'git commit -m "{msg}"', 'git push origin master'): execute(cmd)


class PythonEnv:

    @staticmethod
    def add_path():
        new = ';'.join(map(str, (ENV, ENV / 'Scripts'))) + ';'
        os.environ['PATH'] = new + os.environ['PATH']

    @staticmethod
    def install(pkg, uninstall=False, upgrade=False):
        main = 'uninstall -y' if uninstall else ('install' + upgrade * ' --upgrade')
        execute(f'{pip} {main} --no-cache-dir {pkg}')

    @staticmethod
    def load_requirements(file='requirements.txt'):
        execute(f'{pip} install -r {str(file)} -f https://download.pytorch.org/whl/torch_stable.html')

    @classmethod
    def jupyter(cls, root='D:/Workbench', cfg=False, reinstall=False):
        if reinstall:
            tar = ('jupyte', 'jupyter-client', 'jupyter-console', 'jupyter-core',
                   'jupyterlab-pygments', 'jupyterlab-widgets', 'notebook==6.1.0',
                   'jupyter_contrib_nbextensions')
            for pkg in tar: cls.install(pkg, uninstall=True)
            for pkg in tar: cls.install(pkg)
            execute(f'{jupyter} contrib nbextension install --use')

        os.chdir(root)
        execute(f'{jupyter} notebook' + cfg * ' --generate-config')

    @classmethod
    def config(cls):
        for k, v in (('timeout', 6000),
                     ('index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
                     ('trusted-host', 'pypi.tuna.tsinghua.edu.cn')):
            execute(f'{pip} config set global.{k} {v}')


class CondaEnv(PythonEnv):

    @staticmethod
    def create(name, version=(3, 8, 15)):
        version = '.'.join(map(str, version))
        execute(f'{conda} create -n {name} python=={version}')

    @staticmethod
    def install(pkg, uninstall=False, upgrade=False):
        main = 'uninstall -y' if uninstall else ('upgrade' if upgrade else 'install')
        execute(f'{conda} {main} {pkg}')

    @staticmethod
    def clean():
        execute(f'{conda} clean -ay')

    @classmethod
    def config(cls):
        super().config()
        for p in ('--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
                  '--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
                  '--set show_channel_urls yes'):
            execute(f'{conda} config {p}')


if __name__ == '__main__':
    os.chdir(os.getenv('dl'))

    git_push('D:/Workbench/mod', 'D:/Information/Notes', 'D:/Information/Data/Lib')
