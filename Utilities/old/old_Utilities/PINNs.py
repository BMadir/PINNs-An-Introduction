import torch
import numpy as np
from neural_networks import feedforward

class PINN(feedforward):
    def __init__(self, layers_dim, activations, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

    def grad(self, outputs, inputs):
        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        return grad

    def auto_diff(self, outputs, inputs, n=1):
        if n == 0:
            return outputs
        else:
            grad = self.grad(outputs, inputs)[0]
            return self.auto_diff(grad, inputs, n - 1)

# Navier Stokes velocity pressure formulation
class NavierStokes(PINN):
    def __init__(self, layers_dim, activations, Re, forcing=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

        self.Re = Re
        if forcing is None:
            self.forcing = lambda x, y, t: [0., 0.]
        else:
            self.forcing = lambda x, y, t: forcing(x, y, t, lib=torch)
    @torch.compile
    def net(self, x, y, t):
        outputs = self.forward(torch.cat((x, y, t), dim=1))
        return torch.hsplit(outputs, 3)

    def net_res(self, x, y, t):
        u, v, p = self.net(
            x.requires_grad_(),
            y.requires_grad_(),
            t.requires_grad_()
        )

        u_x, u_y, u_t = self.grad(u, (x, y, t))
        v_x, v_y, v_t = self.grad(v, (x, y, t))
        p_x, p_y = self.grad(p, (x, y))
        u_xx = self.auto_diff(u_x, x)
        u_yy = self.auto_diff(u_y, y)
        v_xx = self.auto_diff(v_x, x)
        v_yy = self.auto_diff(v_y, y)

        f1, f2 = self.forcing(x, y, t)

        res1 = u_x + v_y
        res2 = u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x - f1
        res3 = v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f2

        return res1, res2, res3

    @torch.no_grad()
    def result(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        u, v, p = self.net(x_, y_, t_)

        un = u.detach().cpu().numpy()
        vn = v.detach().cpu().numpy()
        pn = p.detach().cpu().numpy()

        return un, vn, pn

    def grad_p(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        _, _, p = self.net(x_.requires_grad_(), y_.requires_grad_(), t_)

        p_x = self.auto_diff(p, x_).detach().cpu().numpy()
        p_y = self.auto_diff(p, y_).detach().cpu().numpy()
        dp = np.sqrt(p_x ** 2 + p_y ** 2)
        return p_x, p_y, dp


# Navier Stokes velocity vorticity formulation
class NavierStokes_vorticity(PINN):
    def __init__(self, layers_dim, activations, Re, forcing=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

        self.Re = Re
        if forcing is None:
            self.forcing = lambda x, y, t: [0., 0.]
        else:
            self.forcing = lambda x, y, t: forcing(x, y, t, lib=torch)

    def net(self, x, y, t):
        outputs = self.forward(torch.cat((x.requires_grad_(), y.requires_grad_(), t), dim=1))
        stream_fn, p = torch.hsplit(outputs, 2)

        u, v = self.grad(stream_fn, (y, x))
        return u, -v, p

    def net_res(self, x, y, t):
        u, v, p = self.net(
            x.requires_grad_(),
            y.requires_grad_(),
            t.requires_grad_()
        )

        u_x, u_y, u_t = self.grad(u, (x, y, t))
        v_x, v_y, v_t = self.grad(v, (x, y, t))
        p_x, p_y = self.grad(p, (x, y))
        u_xx = self.auto_diff(u_x, x)
        u_yy = self.auto_diff(u_y, y)
        v_xx = self.auto_diff(v_x, x)
        v_yy = self.auto_diff(v_y, y)

        f1, f2 = self.forcing(x, y, t)

        res1 = u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x - f1
        res2 = v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f2

        return res1, res2

    @torch.no_grad()
    def result(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        u, v, p = self.net(x_, y_, t_)

        un = u.detach().cpu().numpy()
        vn = v.detach().cpu().numpy()
        pn = p.detach().cpu().numpy()
        return un, vn, pn

    def grad_p(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        _, _, p = self.net(x_.requires_grad_(), y_.requires_grad_(), t_)

        p_x = self.auto_diff(p, x_).detach().cpu().numpy()
        p_y = self.auto_diff(p, y_).detach().cpu().numpy()
        dp = np.sqrt(p_x ** 2 + p_y ** 2)
        return p_x, p_y, dp

# Natural convection
class Convection(PINN):
    def __init__(self, layers_dim, activations, Re, Pr, Ra, K=1, forcing=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

        self.Re = Re
        self.Pr = Pr
        self.Ra = Ra
        self.K = K
        if forcing is None:
            self.forcing = lambda x, y, t: [0., 0., 0.]
        else:
            self.forcing = lambda x, y, t: forcing(x, y, t, lib=torch)

    def boussinesq_force(self, T):
        return self.Ra/(self.Pr* self.Re**2)* T

    def net(self, x, y, t):
        outputs = self.forward(torch.cat((x, y, t), dim=1))
        return torch.hsplit(outputs, 4)


    def net_res(self, x, y, t):
        u, v, p, T = self.net(
            x.requires_grad_(),
            y.requires_grad_(),
            t.requires_grad_()
        )
        u_x, u_y, u_t = self.grad(u, (x, y, t))
        v_x, v_y, v_t = self.grad(v, (x, y, t))
        T_x, T_y, T_t = self.grad(T, (x, y, t))
        p_x, p_y = self.grad(p, (x, y))

        u_xx = self.auto_diff(u_x, x)
        u_yy = self.auto_diff(u_y, y)
        v_xx = self.auto_diff(v_x, x)
        v_yy = self.auto_diff(v_y, y)
        T_xx = self.auto_diff(T_x, x)
        T_yy = self.auto_diff(T_y, y)

        f1, f2, f3 = self.forcing(x, y, t)
        f_B = self.boussinesq_force(T)
        
        res1 = u_x + v_y
        res2 = u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x - f1
        res3 = v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f_B - f2
        res4 = T_t + u_x * T + u * T_x + v_y * T + v * T_y - self.K * (1/(self.Re* self.Pr)) * (T_xx + T_yy) - f3

        return res1, res4, res2, res3

    @torch.no_grad()
    def result(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        u, v, p, T = self.net(x_, y_, t_)

        un = u.detach().cpu().numpy()
        vn = v.detach().cpu().numpy()
        pn = p.detach().cpu().numpy()
        Tn = T.detach().cpu().numpy()

        return un, vn, pn, Tn

    def grad_p(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        _, _, p, _ = self.net(x_.requires_grad_(), y_.requires_grad_(), t_)

        p_x = self.auto_diff(p, x_).detach().cpu().numpy()
        p_y = self.auto_diff(p, y_).detach().cpu().numpy()
        dp = np.sqrt(p_x ** 2 + p_y ** 2)
        return p_x, p_y, dp

# Phase Change Material (PCM)
class PCM(PINN):
    def __init__(self, layers_dim, activations, Re, Pr, Ra, Ste, K=1, delta=0.05, C=1.e+6, b=1.e-6, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

        self.Re = Re
        self.Pr = Pr
        self.Ra = Ra
        self.Ste = Ste
        self.K = K
        self.delta = delta

        self.C = 1.e+6
        self.b = 1.e-6

    def boussinesq_force(self, T):
        return self.Ra / (self.Pr * self.Re ** 2) * T

    def reg_heaviside(self, T):
        return 0.5* (1 + torch.tanh(T/self.delta))

    def d_reg_heaviside(self, T):
        return (1 / (2 * self.delta)) * (1 - torch.tanh(T/self.delta)** 2)

    def penalty_fct(self, T):
        N = -self.C* (1 - self.reg_heaviside(T))**2
        D = self.reg_heaviside(T)**3 + self.b
        return N/D

    def net(self, x, y, t):
        outputs = self.forward(torch.cat((x, y, t), dim=1))
        return torch.hsplit(outputs, 4)
    
    def net_res(self, x, y, t):
        u, v, p, T = self.net(
            x.requires_grad_(),
            y.requires_grad_(),
            t.requires_grad_()
        )
        u_x, u_y, u_t = self.grad(u, (x, y, t))
        v_x, v_y, v_t = self.grad(v, (x, y, t))
        T_x, T_y, T_t = self.grad(T, (x, y, t))
        p_x, p_y = self.grad(p, (x, y))
        u_xx = self.auto_diff(u_x, x)
        u_yy = self.auto_diff(u_y, y)
        v_xx = self.auto_diff(v_x, x)
        v_yy = self.auto_diff(v_y, y)
        T_xx = self.auto_diff(T_x, x)
        T_yy = self.auto_diff(T_y, y)

        f_B = self.boussinesq_force(T)
        d_phi = self.d_reg_heaviside(T)
        penalty = 1/(1 - self.penalty_fct(T))

        res1 = u_x + v_y
        res2 = penalty* (u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x) + (1 - penalty) * u
        res3 = penalty* (v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f_B) + (1 - penalty) * v
        res4 = T_t + (1/self.Ste)* d_phi* T_t + u_x * T + u * T_x + v_y * T + v * T_y - self.K * (1 / (self.Re * self.Pr)) * (T_xx + T_yy)

        return res1, res2, res3, res4

    @torch.no_grad()
    def result(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        u, v, p, T = self.net(x_, y_, t_)

        un = u.detach().cpu().numpy()
        vn = v.detach().cpu().numpy()
        pn = p.detach().cpu().numpy()
        Tn = T.detach().cpu().numpy()

        return un, vn, pn, Tn


# Navier Stokes velocity grad_pressure formulation
class NavierStokes_dp(PINN):
    def __init__(self, layers_dim, activations, Re, forcing=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

        self.Re = Re
        if forcing is None:
            self.forcing = lambda x, y, t: [0., 0.]
        else:
            self.forcing = lambda x, y, t: forcing(x, y, t, lib=torch)

    def net(self, x, y, t):
        outputs = self.forward(torch.cat((x, y, t), dim=1))
        return torch.hsplit(outputs, 4)

    def net_res(self, x, y, t):
        u, v, p_x, p_y = self.net(
            x.requires_grad_(),
            y.requires_grad_(),
            t.requires_grad_()
        )

        u_x, u_y, u_t = self.grad(u, (x, y, t))
        v_x, v_y, v_t = self.grad(v, (x, y, t))
        u_xx = self.auto_diff(u_x, x)
        u_yy = self.auto_diff(u_y, y)
        v_xx = self.auto_diff(v_x, x)
        v_yy = self.auto_diff(v_y, y)

        p_yx = self.auto_diff(p_y, x)
        p_xy = self.auto_diff(p_x, y)

        f1, f2 = self.forcing(x, y, t)

        res0 = p_xy - p_yx
        res1 = u_x + v_y
        res2 = u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x - f1
        res3 = v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f2

        return res0, res1, res2, res3

    @torch.no_grad()
    def result(self, x, y, t):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        u, v, p_x, p_y = self.net(x_, y_, t_)

        un = u.detach().cpu().numpy()
        vn = v.detach().cpu().numpy()
        p_xn = p_x.detach().cpu().numpy()
        p_yn = p_y.detach().cpu().numpy()
        dpn = np.sqrt(p_xn**2 + p_yn**2)

        return un, vn, dpn

