import pickle
from functools import partial
from pathlib import Path

import optuna


class BayesOpt:
    best = property(lambda self: self.study.best_trial)

    def __init__(self,
                 file: Path,
                 direction: str = "maximize",
                 next_param: dict = None):
        """
        A simple wrapper for Optuna to perform Bayesian optimization.
        :param file: file to save the trials
        :param direction: optimization direction, either "minimize" or "maximize"
        :param next_param: next parameters to enqueue for the optimization
        """
        self.study = optuna.create_study(direction=direction)
        self.file = file
        # load trials from file
        if file.is_file():
            self.study.add_trials(pickle.loads(file.read_bytes()))
        # next parameters
        if next_param: self.enqueue(next_param)
        # function to get the trials dataframe
        self.dataframe = partial(
            self.study.trials_dataframe,
            attrs=("datetime_start", "duration", "params", "value")
        )

    def enqueue(self, params):
        self.study.enqueue_trial(params)
        self.save()

    def save(self):
        self.file.write_bytes(pickle.dumps(self.study.trials))

    def __len__(self):
        return len(tuple(filter(lambda t: t.value, self.study.trials)))

    def __call__(self, func, n_trials=None):
        for i in range(len(self),
                       len(self.study.trials) if n_trials is None else n_trials):
            self.study.optimize(func, 1)
            self.save()


if __name__ == "__main__":
    def obj(trial: optuna.Trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x ** 2 + y


    opt = BayesOpt(Path("test.bin"))
    opt(obj, 10)
