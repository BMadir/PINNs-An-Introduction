# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
import numpy as np
from torch import Tensor

from torch.optim.optimizer import Optimizer, ParamsT
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from scipy.sparse.linalg import cg
from scipy.linalg import lu_factor, lu_solve

__all__ = ["Newton"]

class Newton(Optimizer):

    def __init__(
        self,
        params: ParamsT,
        solver: Optional[str] = "cg",
        tolerance: float = 1e-5,
    ):
        defaults = dict(
            solver=solver,
            tolerance=tolerance,
        )
        super().__init__(params, defaults)
        self._params = self.param_groups[0]["params"]

    @staticmethod
    def grad_fn(func, params):
        if not torch.is_tensor(params):
            params = torch.Tensor(params)
        grad = torch.autograd.functional.jacobian(func, params)
        return grad.reshape(-1)

    @staticmethod
    def hessian_fn(func, params):
        if not torch.is_tensor(params):
            params = torch.Tensor(params)
        hessian = torch.autograd.functional.hessian(func, params)
        return hessian

    @torch.no_grad()
    def step(self, closure):

        group = self.param_groups[0]
        solver = group["solver"]
        tolerance = group["tolerance"]

        state = self.state[self._params[0]]
        params = parameters_to_vector(self._params)
        loss = closure(params)

        H = self.hessian_fn(closure, params)
        g = self.grad_fn(closure, params)

        if solver == "lu":
            lu, piv = lu_factor(H.numpy())
            d = lu_solve((lu, piv), -g.numpy())
        else:
            d, _ = cg(H.numpy(), -g.numpy(), rtol=1e-7)

        r_norm = np.linalg.norm(H.numpy() @ d + g.numpy())
        if r_norm > tolerance:
            exit_code = 1
            print(f"{solver.upper()} didn't converge, r_norm: {r_norm:.3e}")
        else:
            exit_code = 0
            params += torch.Tensor(d)
            vector_to_parameters(params, self._params)
            loss = closure(params)

        return loss, exit_code