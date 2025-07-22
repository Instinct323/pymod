import casadi

import numpy as np


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

    def optimize(self) -> list[np.ndarray]:
        """ Perform the optimization and return the optimized variable values """
        self.minimize(self.total_loss())
        try:
            self.solve()
            vget = self.value
        except RuntimeError as e:
            if self.print_level > 0: print(f"Failed to optimize: {e}")
            vget = self.debug.value
        return [vget(var) for var in self.__vars]
