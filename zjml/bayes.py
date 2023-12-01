import pickle
from functools import partial
from pathlib import Path

import optuna


class BayerOpt:
    best = property(lambda self: self.study.best_trial)

    def __init__(self,
                 file: Path,
                 direction: str = 'maximize',
                 nextp: dict = None):
        self.study = optuna.create_study(direction=direction)
        self.file = file
        # 加载已完成的试验
        if file.is_file():
            self.study.add_trials(pickle.loads(file.read_bytes()))
        # 加载一组指定参数
        if nextp:
            self.study.enqueue_trial(nextp)
        # 函数重命名
        self.dataframe = partial(
            self.study.trials_dataframe,
            attrs=('datetime_start', 'duration', 'params', 'value')
        )

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
