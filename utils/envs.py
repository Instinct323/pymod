import datetime
import os
from pathlib import Path

PYPATH = Path('D:/Workbench/Library/envs/cv')
CONDAPATH = Path('D:/Software/Anaconda3')
WORKDIR = Path('D:/Workbench')


def git_push(*repositories,
             msg=f'update on {datetime.datetime.today()}'):
    for repo in repositories:
        os.chdir(repo), print(repo.center(50, '-'))
        for cmd in ('git status', 'git add .',
                    f'git commit -m "{msg}"', 'git push'): os.system(cmd)


class PythonEnv:

    @staticmethod
    def install(pkg, uninstall=False, upgrade=False):
        main = 'uninstall -y' if uninstall else ('install' + upgrade * ' --upgrade')
        os.system(f'pip {main} --no-cache-dir {pkg}')

    @staticmethod
    def load_requirements(file='requirements.txt'):
        os.system(f'pip install -r {str(file)} -f https://download.pytorch.org/whl/torch_stable.html')

    @classmethod
    def jupyter(cls, root=WORKDIR, cfg=False, reinstall=False):
        if reinstall:
            tar = ('jupyter', 'jupyter-client', 'jupyter-console', 'jupyter-core',
                   'jupyterlab-pygments', 'jupyterlab-widgets', 'notebook==6.1.0',
                   'jupyter_contrib_nbextensions')
            for pkg in tar: cls.install(pkg, uninstall=True)
            for pkg in tar: cls.install(pkg)
            os.system('jupyter contrib nbextension install --user')

        os.chdir(str(root))
        os.system('jupyter notebook' + cfg * ' --generate-config')

    @staticmethod
    def config():
        for k, v in (('timeout', 6000),
                     ('index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
                     ('trusted-host', 'pypi.tuna.tsinghua.edu.cn')):
            os.system(f'pip config set global.{k} {v}')

    @staticmethod
    def modify_env(env_path: Path = PYPATH):
        os.environ['PATH'] = ';'.join(map(
            str, [env_path, env_path / 'bin', env_path / 'Scripts',
                  env_path / 'Library/bin', env_path / 'Library/usr/bin', env_path / 'Library/mingw-w64/bin',
                  ''])) + os.environ['PATH']


class CondaEnv(PythonEnv):

    @staticmethod
    def create(name, version=(3, 8, 0)):
        version = '.'.join(map(str, version))
        os.system(f'conda create -n {name} python=={version}')

    @staticmethod
    def install(pkg, uninstall=False, upgrade=False):
        main = 'uninstall -y' if uninstall else ('upgrade' if upgrade else 'install')
        os.system(f'conda {main} {pkg}')

    @staticmethod
    def clean():
        os.system('conda clean -ay')

    @staticmethod
    def config():
        super().config()
        for p in ('--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
                  '--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
                  '--set show_channel_urls yes'):
            os.system(f'conda config {p}')

    @classmethod
    def modify_env(cls, env_path: Path = PYPATH,
                   conda_path: Path = CONDAPATH):
        super().modify_env(env_path)
        import warnings
        warnings.warn('This function is incomplete', DeprecationWarning)

        os.environ['CONDA_DEFAULT_ENV'] = str(env_path)
        os.environ['CONDA_PREFIX'] = str(env_path)
        os.environ['CONDA_PROMPT_MODIFIER'] = f'({env_path})'
        os.environ['CONDA_SHLVL'] = '1'
        os.environ['PROMPT'] = f'({env_path}) $P$G'

        os.environ['PATH'] = ';'.join(map(
            str, [conda_path / 'condabin', conda_path / 'Library/Bin', ''])) + os.environ['PATH']


if __name__ == '__main__':
    os.chdir(os.getenv('dl'))
    PythonEnv.modify_env()

    env = PythonEnv()
    # env.load_requirements(r'D:\Information\Python\mod\requirements.txt')
    git_push(r'D:\Workbench\mod', r'D:\Information\Notes', r'D:\Information\Notes\info', r'D:\Workbench\Library')
    # env.jupyter()
