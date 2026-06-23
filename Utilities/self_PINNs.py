import torch
import numpy as np
from neural_networks import feedforward
from AutoDiff import AG_grad, args_requires_grad

__all__ = ["Stefan_pinn"]

class Stefan_pinn(feedforward):
    def __init__(self, layers_dim, activations, Ste, Fo, delta, dim=1, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)
        self.Ste = Ste
        self.Fo = Fo
        self.delta = delta
        self.dim = dim

    def net(self, *args):
        inputs = torch.cat(args, dim=1)
        outputs = self.__call__(inputs)
        T, mu = torch.hsplit(outputs, 2)
        return T, mu

    @staticmethod
    def phi(T, delta):
        return (1 / 2) * (1 + torch.tanh(T / delta))

    @staticmethod
    def phi_der(T, delta):
        return (1 / (2 * delta)) * (1 - torch.tanh(T / delta) ** 2)

    @args_requires_grad
    def net_grad(self, *args):
        T, mu = self.net(*args)
        grad = AG_grad(T, args)
        return *grad, mu

    @args_requires_grad
    def net_res(self, *args):
        if self.dim == 1:
            return self.net_res_1D(*args)
        elif self.dim == 2:
            return  self.net_res_2D(*args)
        else:
            raise NotImplementedError(f"Stefan pb for dim = {self.dim} is not implemented")

    def net_res_1D(self, t, x):
        T, mu = self.net(t, x)
        Tt, Tx = AG_grad(T, (t, x))
        Txx = AG_grad(Tx, x)
        res = (1. + (1. / self.Ste) * self.phi_der(T, self.delta)) * Tt - (self.Fo) * Txx
        return res, mu
    
    def net_res_2D(self, t, x, y):
        T, mu = self.net(t, x, y)
        Tt = AG_grad(T, t)
        Txx, Tyy = AG_grad(T, (x, y), order=2)
        res = (1. + (1. / self.Ste) * self.phi_der(T, self.delta)) * Tt - (self.Fo) * (Txx + Tyy)
        return res, mu

    def evaluate(self, *args):
        inputs = list()
        for arg in args:
            if not torch.is_tensor(arg):
                arg = torch.Tensor(arg)
            inputs.append(arg.to(self.device))
        outputs = self.net(*inputs)
        if torch.is_tensor(outputs):
            return outputs.detach().cpu().numpy()
        else:
            return tuple(out.detach().cpu().numpy() for out in outputs)