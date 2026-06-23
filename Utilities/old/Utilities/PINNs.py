import torch
import numpy as np
from neural_networks import feedforward
from AutoDiff import AG_grad, args_requires_grad
from utils import  evalute


__all__ = [
    "NC_pinn", "NC_vv_pinn"]

# Velocity Pressure (VP) formulation
class NC_pinn(feedforward):
    def __init__(self, layers_dim, activations, Re, Pr, Ra, K, dim=2, rhs=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)
        assert layers_dim[-1] == self.dim + 2, "error in output dim"
        self.Re = Re
        self.Pr = Pr
        self.Ra = Ra
        self.K = K
        self.dim = dim
        
        if callable(rhs):
            F = rhs
        else:
            F = lambda *args: len(args) * [0.]
        self.F = F
        
    def net(self, *args):
        inputs = torch.cat(args, dim=1)
        outputs = self.__call__(inputs)
        return torch.hsplit(outputs, self.dim + 2)
    
    def f_B(self, T):
        cst = self.Ra/(self.Pr * self.Re**2)
        return cst * T
    
    @args_requires_grad
    def net_res(self, *args):
        if self.dim == 2:
            return self.net_res_2D(*args)
        else:
            raise NotImplementedError(f"natural convection pb for dim = {self.dim} is not implemented")

    def net_res_2D(self, t, x, y):
        u1, u2, p, T = self.net(t, x, y)
        # 1 der
        u1_t, u1_x, u1_y = AG_grad(u1, (t, x, y))
        u2_t, u2_x, u2_y = AG_grad(u2, (t, x, y))
        T_t, T_x, T_y = AG_grad(T, (t, x, y))
        p_x, p_y = AG_grad(p, (x, y))
        # 2 der
        u1_xx = AG_grad(u1_x, x)
        u1_yy = AG_grad(u1_y, y)
        u2_xx = AG_grad(u2_x, x)
        u2_yy = AG_grad(u2_y, y)
        T_xx = AG_grad(T_x, x)
        T_yy = AG_grad(T_y, y)
        # rhs
        F1, F2, F3 = self.F(t, x, y)
        # res
        res1 = u1_x + u2_y
        res2 = u1_t + u1 * u1_x + u2 * u1_y - (1. / self.Re) * (u1_xx + u1_yy) + p_x - F1
        res3 = u2_t + u1 * u2_x + u2 * u2_y - (1. / self.Re) * (u2_xx + u2_yy) + p_y - self.f_B(T) - F2
        res4 = T_t + u1_x * T + u1 * T_x + u2_y * T + u2 * T_y - self.K * (1/(self.Re* self.Pr)) * (T_xx + T_yy) - F3
        return res1, res2, res3, res4

    @args_requires_grad
    def net_grad(self, *args, index=-1):
        outputs = self.net(*args)
        grad = AG_grad(outputs[index], args)
        return grad

# Velocity Vorticity (VV) formulation
class NC_vv_pinn(feedforward):
    def __init__(self, layers_dim, activations, Re, Pr, Ra, K, dim=2, rhs=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)
        assert layers_dim[-1] == 3, "error in output dim"
        self.Re = Re
        self.Pr = Pr
        self.Ra = Ra
        self.K = K
        self.dim = dim

        if callable(rhs):
            F = rhs
        else:
            F = lambda *args: len(args) * [0.]
        self.F = F

    @args_requires_grad
    def net(self, *args):
        if self.dim == 2:
            return self.net_2D(*args)
        else:
            raise NotImplementedError(f"natural convection pb for dim = {self.dim} is not implemented")

    def net_2D(self, *args):
        inputs = torch.cat(args, dim=1)
        outputs = self.__call__(inputs)
        stream_fn, p, T = torch.hsplit(outputs, 3)
        u, v = AG_grad(stream_fn, (y, x))
        return u, -v, p

    def f_B(self, T):
        cst = self.Ra/(self.Pr * self.Re**2)
        return cst * T

    @args_requires_grad
    def net_res(self, *args):
        if self.dim == 2:
            return self.net_res_2D(*args)
        else:
            raise NotImplementedError(f"natural convection pb for dim = {self.dim} is not implemented")

    def net_res_2D(self, t, x, y):
        u1, u2, p, T = self.net(t, x, y)
        # 1 der
        u1_t, u1_x, u1_y = AG_grad(u1, (t, x, y))
        u2_t, u2_x, u2_y = AG_grad(u2, (t, x, y))
        T_t, T_x, T_y = AG_grad(T, (t, x, y))
        p_x, p_y = AG_grad(p, (x, y))
        # 2 der
        u1_xx = AG_grad(u1_x, x)
        u1_yy = AG_grad(u1_y, y)
        u2_xx = AG_grad(u2_x, x)
        u2_yy = AG_grad(u2_y, y)
        T_xx = AG_grad(T_x, x)
        T_yy = AG_grad(T_y, y)
        # rhs
        F1, F2, F3 = self.F(t, x, y)
        # res
        res1 = u1_t + u1 * u1_x + u2 * u1_y - (1. / self.Re) * (u1_xx + u1_yy) + p_x - F1
        res2 = u2_t + u1 * u2_x + u2 * u2_y - (1. / self.Re) * (u2_xx + u2_yy) + p_y - self.f_B(T) - F2
        res3 = T_t + u1_x * T + u1 * T_x + u2_y * T + u2 * T_y - self.K * (1 / (self.Re * self.Pr)) * (T_xx + T_yy) - F3
        return res1, res2, res3

    @args_requires_grad
    def net_grad(self, *args, index=-1):
        outputs = self.net(*args)
        grad = AG_grad(outputs[index], args)
        return grad