import casadi

import numpy as np


def mean(mat: casadi.MX,
         axis: int = None) -> casadi.MX:
    if axis is None:
        return casadi.sum(mat) / mat.numel()
    elif axis == 0:
        return casadi.sum1(mat) / mat.shape[0]
    elif axis in (1, -1):
        return casadi.sum2(mat) / mat.shape[1]
    else:
        raise NotImplementedError


def variance(mat: casadi.MX,
             axis: int = None) -> casadi.MX:
    un_bias = mat - broadcast_to(mean(mat, axis=axis), mat.shape)
    return mean(un_bias ** 2, axis=axis)


def diff(mat: casadi.MX,
         axis: int) -> casadi.MX:
    if axis == 0:
        return mat[1:, :] - mat[:-1, :]
    elif axis == 1:
        return mat[:, 1:] - mat[:, :-1]
    else:
        raise NotImplementedError


def norm(mat: casadi.MX,
         axis: int = None) -> casadi.MX:
    if axis is None:
        return casadi.norm_fro(mat)
    elif axis == 0:
        return casadi.hcat([casadi.norm_2(mat[:, i]) for i in range(mat.shape[1])])
    elif axis in (1, -1):
        return casadi.vcat([casadi.norm_2(mat[i, :]) for i in range(mat.shape[0])])
    else:
        raise NotImplementedError


def broadcast_to(vec: casadi.MX,
                 shape: tuple[int, int]) -> casadi.MX:
    repeat = [(s if v == 1 else 1) for v, s in zip(vec.shape, shape)]
    return casadi.repmat(vec, *repeat)


class CasADiOpti(casadi.Opti):

    def __init__(self,
                 max_iter: int,
                 tol: float,
                 print_level: int = 5,
                 print_time: bool = True):
        super().__init__()
        assert (max_iter + 1) % 10 == 0
        self.print_level = print_level
        self.solver("ipopt", {
            "ipopt.max_iter": max_iter, "ipopt.tol": tol,
            "ipopt.print_level": print_level, "print_time": print_time
        })
        self.__vars = []
        self.__constraints = []
        self.__losses = []
        self.__weights = []
        self.__ckpts = []

    def variable(self,
                 shape: tuple[int, int],
                 initial: np.ndarray = None) -> casadi.MX:
        """ Create a variable with the given shape and optional initial value """
        var = super().variable(*shape)
        if initial is not None: self.set_initial(var, initial)
        self.__vars.append(var)
        return var

    def add_constraint(self,
                       constraint: casadi.MX):
        """ Add a constraint to the optimization problem """
        self.__constraints.append(constraint)
        self.subject_to(constraint)

    def add_loss(self,
                 loss: casadi.MX,
                 weight: float = 1.):
        """ Add a loss term to the optimization problem with an optional weight """
        self.__losses.append(loss)
        self.__weights.append(weight)

    def add_ckpt(self,
                 ckpt: casadi.MX):
        """ Add a checkpoint to the optimization problem """
        self.__ckpts.append(ckpt)

    def total_loss(self) -> casadi.MX:
        """ Calculate the total loss as a weighted sum of individual losses """
        assert len(self.__losses)
        return sum(w * loss for w, loss in zip(self.__weights, self.__losses))

    def func_constraint(self, name: str = "constraint"):
        """ Create a CasADi function for the constraints """
        return casadi.Function(name, self.__vars, self.__constraints)

    def func_loss(self, name: str = "loss"):
        """ Create a CasADi function for the losses """
        return casadi.Function(name, self.__vars, self.__losses + [self.total_loss()])

    def func_ckpt(self, name: str = "ckpt"):
        return casadi.Function(name, self.__vars, self.__ckpts)

    def optimize(self) -> list[np.ndarray]:
        """ Perform the optimization and return the optimized variable values """
        self.minimize(self.total_loss())
        try:
            self.solve()
            vget = self.value
        except RuntimeError as e:
            if self.print_level > 0: print(f"Failed to optimize: {e}")
            vget = self.debug.value
        return [(v[:, None] if v.ndim == 1 else v) for v in (vget(var) for var in self.__vars)]


if __name__ == '__main__':
    a = casadi.DM(np.arange(12).reshape(3, 4))
    print(a)
    print(variance(a, axis=None))
