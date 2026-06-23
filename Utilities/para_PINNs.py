import torch
import numpy as np
from neural_networks import feedforward, ResNet, sf_net, ff_net
from AutoDiff import AG_grad, args_requires_grad


from scipy.optimize import newton
import scipy.special as ss

__all__ = ["Stefan"]

class Stefan(feedforward):
    def __init__(self, layers_dim, activations, Ste, Fo, delta, dim=1, mu=0., sigma=1., **kwargs):
        super().__init__(layers_dim, activations, **kwargs)
        self.Ste = Ste
        self.Fo = Fo
        self.delta = delta
        self.dim = dim
        self.mu = mu
        self.sigma = sigma

        self.t0 = 0.05
        self.Th = 1.
        self.Tc = -0.5

    def _net(self, *args):
        inputs = torch.cat(args, dim=1)
        inputs_ = (inputs - self.mu)/self.sigma
        outputs = self.__call__(inputs_)
        return outputs

    @staticmethod
    def phi(T, delta):
        return (1 / 2) * (1 + torch.tanh(T / delta))

    @staticmethod
    def phi_der(T, delta):
        return (1 / (2 * delta)) * (1 - torch.tanh(T / delta) ** 2)

    def _T0(self, x, Ste):
        if torch.is_tensor(Ste):
            Ste_np = Ste.detach().cpu().numpy()
        else:
            Ste_np = Ste
        Fo = self.Fo
        t0 = torch.tensor(self.t0)
        Th = self.Th
        Tc = self.Tc

        f = lambda x: x - (Ste_np / np.sqrt(np.pi)) * np.exp(- x ** 2) * (Tc / ss.erfc(x) + Th / ss.erf(x))
        x0 = 0.01 * np.ones((len(Ste), 1))
        lam = newton(f, x0)
        lam = torch.Tensor(lam).to(self.device)

        S = 2 * lam * torch.sqrt(Fo * t0).to(self.device)
        N = torch.erf(x / (2 * torch.sqrt(Fo * t0))).to(self.device)
        D = torch.erf(lam).to(self.device)
        Tl = Th * (1 - N / D)
        Ts = Tc * (1 - (1 - N) / (1 - D))
        return torch.where(x < S, Tl, Ts)

    def net(self, t, x, Ste):
        return self._T0(x, Ste) + (t - self.t0)* self._net(t, x, Ste)

    @args_requires_grad
    def net_grad(self, *args):
        T = self.net(*args)
        grad = AG_grad(T, args)
        return grad

    @args_requires_grad
    def net_res(self, *args):
        if self.dim == 1:
            return self.net_res_1D(*args)
        elif self.dim == 2:
            return  self.net_res_2D(*args)
        else:
            raise NotImplementedError(f"Stefan pb for dim = {self.dim} is not implemented")

    def net_res_1D(self, t, x, Ste):
        T = self.net(t, x, Ste)
        Tt, Tx = AG_grad(T, (t, x))
        Txx = AG_grad(Tx, x)
        res = (1. + (1. / Ste) * self.phi_der(T, self.delta)) * Tt - (self.Fo) * Txx
        return res
    
    def net_res_2D(self, t, x, y):
        T = self.net(t, x, y)
        Tt = AG_grad(T, t)
        Txx, Tyy = AG_grad(T, (x, y), order=2)
        res = (1. + (1. / self.Ste) * self.phi_der(T, self.delta)) * Tt - (self.Fo) * (Txx + Tyy)
        return res

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