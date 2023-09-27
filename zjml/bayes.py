import pickle
from pathlib import Path

import optuna


class BayerOpt:

    def __init__(self,
                 file: Path,
                 direction: str = 'maximize',
                 best: dict = None):
        self.study = optuna.create_study(direction=direction,
                                         storage='mysql://zjtong:20010323@localhost:3306/pyresult')
        self.file = file
        self.best = best
        # 加载已完成的试验, 如果文件不存在则尝试最优参数
        '''if file.is_file():
            self.study.add_trials(pickle.loads(file.read_bytes()))
        elif best:
            self.study.enqueue_trial(best)'''

    def __call__(self, func, n_trials):
        for i in range(n_trials):
            self.study.optimize(func, 1)
            self.file.write_bytes(pickle.dumps(self.study.trials))
            # print('Automatically saved.')


if __name__ == '__main__':
    def obj(trial: optuna.Trial):
        x = trial.suggest_float('x', 0, 1)
        y = trial.suggest_float('y', 0, 1)
        return x ** 2 + y


    opt = BayerOpt(Path('test.bin'))
    opt(obj, 10)
