# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor

from torch.optim.optimizer import Optimizer, ParamsT
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from scipy.optimize import minimize

__all__ = ["Newton"]

class Newton(Optimizer):

    def __init__(
        self,
        params: ParamsT,
        max_iter: int = None,
        tolerance: float = 1e-9,
    ):
        defaults = dict(
            max_iter=max_iter,
            tolerance=tolerance,
        )
        super().__init__(params, defaults)
        self._params = self.param_groups[0]["params"]

    """
    @staticmethod
    def _grad_hessian_fn(func, argnums=0):
        grad_fn = torch.func.jacrev(func, argnums)
        hessian_fn = torch.func.jacfwd(grad_fn, argnums)
        return grad_fn, hessian_fn
    """

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
        max_iter = group["max_iter"]
        tolerance = group["tolerance"]

        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("jacobian_evals", 0)
        state.setdefault("hessian_evals", 0)
        state.setdefault("n_iter", 0)

        #grad_fn, hessian_fn = self._grad_hessian_fn(closure)

        method = "Newton-CG"
        fn = closure
        jac_fn = lambda params: self.grad_fn(func=closure, params=params)
        hess_fn = lambda params: self.hessian_fn(func=closure, params=params)
        params = parameters_to_vector(self._params)
        options = dict(
            disp=False,
            maxiter=max_iter,
            return_all=True
        )

        result = minimize(
            fun=fn,
            x0=params,
            method=method,
            jac=jac_fn,
            hess=hess_fn,
            tol=tolerance,
            options=options
        )

        params_opt = torch.Tensor(result.x)
        vector_to_parameters(params_opt, self._params)

        loss = torch.Tensor([result.fun])

        n_iter = result.nit
        fn_eval, grad_fn_eval, hessian_fn_eval = result.nfev, result.njev, result.nhev
        jacobian = result.jac

        state["n_iter"] += n_iter

        state["func_evals"] += fn_eval
        state["jacobian_evals"] += grad_fn_eval
        state["hessian_evals"] += hessian_fn_eval

        state["jacobian"] = torch.Tensor(jacobian)
        return loss
