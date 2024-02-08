import datetime
import os
from pathlib import Path

from zjexe import execute


def add_path():
    sep = ';' if os.name == 'nt' else ':'
    env_path = os.environ['PATH'].split(sep)

    # Window
    if os.name == 'nt':
        env = Path('D:/Software/envs/cv')
        conda = Path('D:/Software/Anaconda3/condabin')
        ext_path = [env, conda, env / 'Scripts']

    # Linux
    else:
        env = Path('/home/slam602/.conda/envs/torch/bin')
        conda = Path('/opt/miniconda/bin')
        ext_path = [env, conda]

    ext_path = [str(p) for p in ext_path if str(p) not in env_path]
    os.environ['PATH'] = sep.join(ext_path + env_path)
    return ext_path


def git_push(*repositories,
             msg=f'update on {datetime.datetime.today()}'):
    for repo in repositories:
        os.chdir(repo), print(repo.center(50, '-'))
        for cmd in ('git status', 'git add .',
                    f'git commit -m "{msg}"', 'git push origin master'): execute(cmd, check=False)


class PythonEnv:

    @staticmethod
    def install(pkg, uninstall=False, upgrade=False):
        main = 'uninstall -y' if uninstall else ('install' + upgrade * ' --upgrade')
        execute(f'pip {main} --no-cache-dir {pkg}')

    @staticmethod
    def load_requirements(file='requirements.txt'):
        execute(f'pip install -r {str(file)} -f https://download.pytorch.org/whl/torch_stable.html')

    @classmethod
    def jupyter(cls, root='D:/Workbench', cfg=False, reinstall=False):
        if reinstall:
            tar = ('jupyte', 'jupyter-client', 'jupyter-console', 'jupyter-core',
                   'jupyterlab-pygments', 'jupyterlab-widgets', 'notebook==6.1.0',
                   'jupyter_contrib_nbextensions')
            for pkg in tar: cls.install(pkg, uninstall=True)
            for pkg in tar: cls.install(pkg)
            execute(f'jupyter contrib nbextension install --use')

        os.chdir(root)
        execute(f'jupyter notebook' + cfg * ' --generate-config')

    @classmethod
    def config(cls):
        for k, v in (('timeout', 6000),
                     ('index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
                     ('trusted-host', 'pypi.tuna.tsinghua.edu.cn')):
            execute(f'pip config set global.{k} {v}')


class CondaEnv(PythonEnv):

    @staticmethod
    def create(name, version=(3, 8, 15)):
        version = '.'.join(map(str, version))
        execute(f'conda create -n {name} python=={version}')

    @staticmethod
    def install(pkg, uninstall=False, upgrade=False):
        main = 'uninstall -y' if uninstall else ('upgrade' if upgrade else 'install')
        execute(f'conda {main} {pkg}')

    @staticmethod
    def clean():
        execute('conda clean -ay')

    @classmethod
    def config(cls):
        super().config()
        for p in ('--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
                  '--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
                  '--set show_channel_urls yes'):
            execute(f'conda config {p}')


if __name__ == '__main__':
    add_path()
    os.chdir(os.getenv('dl'))

    git_push('D:/Workbench/cppmod', 'D:/Workbench/pymod', 'D:/Information/Notes', 'D:/Information/Lib')
