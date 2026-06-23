import torch
from neural_networks import*

from itertools import chain

class PINN(feedforward):

    def __init__(self, layers_dim, activations, Re, forcing=None, **kwargs):
        super().__init__(layers_dim, activations, **kwargs)

        self.Re = Re
        self.forcing = forcing

    def F(self, x, y, t):
        if self.forcing is None:
            return [0., 0.]
        else:
            return self.forcing(x, y, t, lib=torch)

    def net(self, x, y, t):
        uvp = self.forward(torch.cat((x, y, t), dim=1))
        return torch.hsplit(uvp, 3)

    def net_res(self, x, y, t):

        if not t.requires_grad:
            t.requires_grad = True

        if not x.requires_grad:
            x.requires_grad = True

        if not y.requires_grad:
            y.requires_grad = True

        u, v, p = self.net(x, y, t)

        u_t = self.auto_diff(u, t)
        v_t = self.auto_diff(v, t)

        u_x = self.auto_diff(u, x)
        u_xx = self.auto_diff(u_x, x)

        v_x = self.auto_diff(v, x)
        v_xx = self.auto_diff(v_x, x)

        u_y = self.auto_diff(u, y)
        u_yy = self.auto_diff(u_y, y)

        v_y = self.auto_diff(v, y)
        v_yy = self.auto_diff(v_y, y)

        p_x = self.auto_diff(p, x)
        p_y = self.auto_diff(p, y)

        f1, f2 = self.F(x, y, t)

        res1 = u_x + v_y
        res2 = u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x - f1
        res3 = v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f2

        return res1, res2, res3

    def result(self, x, y, t, grad_p=False):
        x_ = torch.Tensor(x).to(self.device)
        y_ = torch.Tensor(y).to(self.device)
        t_ = torch.Tensor(t).to(self.device)

        u, v, p = self.net(x_, y_, t_)

        if grad_p:
            p_x = self.auto_diff(p, x)
            p_y = self.auto_diff(p, y)
            return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy(), p_x.detach().cpu().numpy(), p_y.detach().cpu().numpy()
        else:
            return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()

    def auto_diff(self, y, x, n=1):
        if n == 0:
            return y
        else:
            dy_dx = torch.autograd.grad(y, x,
                                        torch.ones_like(y),
                                        create_graph=True,
                                        retain_graph=True,
                                        allow_unused=True)[0]
            return self.auto_diff(dy_dx, x, n - 1)

    def save(self, name='model', file=''):
        save_dict = {'layers_dim': self.layers_dim,
                     'activations': self.activations,
                     'state_dict': self.state_dict()}

        torch.save(save_dict, file + name + '.pth')

class PINN_phi(feedforward):
    
    def __init__(self, layers_dim, activations, Re, forcing=None, **kwargs) :
        super().__init__(layers_dim, activations, **kwargs)
        
        self.Re = Re
        self.forcing = forcing

    def F(self, x, y, t):
        if self.forcing is None:
            return [0., 0.]
        else:
            return self.forcing(x, y, t, lib=torch)

    def net(self, x, y, t):
        
        if not x.requires_grad:
            x.requires_grad = True
        
        if not y.requires_grad:
            y.requires_grad = True
            
        psip = self.forward(torch.cat((x, y, t), dim = 1))
        psi, p = torch.hsplit(psip, 2)
        
        u = self.auto_diff(psi, y)
        v = -1.* self.auto_diff(psi, x)
        return u, v, p
    
    def net_res(self, x, y, t):
        
        if not t.requires_grad:
            t.requires_grad = True
        
        if not x.requires_grad:
            x.requires_grad = True
        
        if not y.requires_grad:
            y.requires_grad = True
        
        u, v, p = self.net(x, y, t)
        
        u_t = self.auto_diff(u, t)
        v_t = self.auto_diff(v, t)
        
        u_x = self.auto_diff(u, x)
        u_xx = self.auto_diff(u_x, x)
        
        v_x = self.auto_diff(v, x)
        v_xx = self.auto_diff(v_x, x)
        
        u_y = self.auto_diff(u, y)
        u_yy = self.auto_diff(u_y, y)
        
        v_y = self.auto_diff(v, y)
        v_yy = self.auto_diff(v_y, y)
        
        p_x = self.auto_diff(p, x)
        p_y = self.auto_diff(p, y)

        f1, f2 = self.F(x, y, t)
        res1 = u_t + u * u_x + v * u_y - (1./self.Re) * (u_xx + u_yy) + p_x - f1
        res2 = v_t + u * v_x + v * v_y - (1./self.Re) * (v_xx + v_yy) + p_y - f2
        
        return res1, res2
    
    def result(self, x, y, t, grad_p = False):    
        x_  = torch.Tensor(x).to(self.device)
        y_  = torch.Tensor(y).to(self.device)
        t_  = torch.Tensor(t).to(self.device)
        
        u, v, p = self.net(x_, y_, t_)
        
        if grad_p:
            p_x = self.auto_diff(p, x)
            p_y = self.auto_diff(p, y)
            return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy(), p_x.detach().cpu().numpy(), p_y.detach().cpu().numpy()
        else:
            return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()
    
    def auto_diff(self, y, x, n = 1):
        if n == 0:
            return y
        else:
            dy_dx = torch.autograd.grad(y, x, 
                                        torch.ones_like(y), 
                                        create_graph=True, 
                                        retain_graph=True, 
                                        allow_unused=True)[0]
            return self.auto_diff(dy_dx, x, n - 1)
        
    def save(self, name = 'model', file = ''):
        save_dict = {'layers_dim'  : self.layers_dim, 
                     'activations' : self.activations,
                     'state_dict'  : self.state_dict()}
        
        torch.save(save_dict, file + name + '.pth')


class MPINN:
    
    def __init__(self, PINN1, PINN2, Re, forcing=None):
        
        assert PINN1.device == PINN2.device, ('check pinns devices')
        self.device = PINN1.device
        
        self.PINNs = [PINN1, PINN2]
        
        self.net1 = PINN1.net
        self.net2 = PINN2.net
        
        self.Re = Re
        self.forcing = forcing
    
    def parameters(self):
        g = [pinn.parameters() for pinn in self.PINNs]
        return chain(*g)
    
    def named_parameters(self):
        g = [pinn.named_parameters(prefix='pinn_%i'%i) for i, pinn in enumerate(self.PINNs)]
        return chain(*g)
    
    def zero_grad(self):
        for pinn in self.PINNs:
            pinn.zero_grad()

    def F(self, x, y, t):
        if self.forcing is None:
            return [0., 0.]
        else:
            return self.forcing(x, y, t, lib=torch)

    def net(self, x, y, t):
        return *self.net1(x, y, t), *self.net2(x, y, t)
    
    def net_res(self, x, y, t):
        
        if not t.requires_grad:
            t.requires_grad = True
        
        if not x.requires_grad:
            x.requires_grad = True
        
        if not y.requires_grad:
            y.requires_grad = True
        
        u, v = self.net1(x, y, t)
        p, = self.net2(x, y, t)
        
        u_t = self.auto_diff(u, t)
        v_t = self.auto_diff(v, t)

        u_x = self.auto_diff(u, x)
        u_xx = self.auto_diff(u_x, x)

        v_x = self.auto_diff(v, x)
        v_xx = self.auto_diff(v_x, x)

        u_y = self.auto_diff(u, y)
        u_yy = self.auto_diff(u_y, y)

        v_y = self.auto_diff(v, y)
        v_yy = self.auto_diff(v_y, y)

        p_x = self.auto_diff(p, x)
        p_y = self.auto_diff(p, y)

        f1, f2 = self.F(x, y, t)
        res1 = u_x + v_y
        res2 = u_t + u * u_x + v * u_y - (1. / self.Re) * (u_xx + u_yy) + p_x - f1
        res3 = v_t + u * v_x + v * v_y - (1. / self.Re) * (v_xx + v_yy) + p_y - f2
        
        return res1, res2, res3
    
    def auto_diff(self, y, x, n = 1):
        if n == 0:
            return y
        else:
            dy_dx = torch.autograd.grad(y, x, 
                                        torch.ones_like(y), 
                                        create_graph=True, 
                                        retain_graph=True, 
                                        allow_unused=True)[0]
            return self.auto_diff(dy_dx, x, n - 1)
    
    def result(self, x, y, t):    
        x_  = torch.Tensor(x).to(self.device)
        y_  = torch.Tensor(y).to(self.device)
        t_  = torch.Tensor(t).to(self.device)
        
        u, v, p = self.net(x_, y_, t_)
        return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()

"""layers = [3, 20, 3]
activations = 'tanh'
Re = 100
Navier_Stokes = PINN(layers, activations, Re, [0, 0])

a = torch.rand(1, 1).cuda()
b = Navier_Stokes.net(a, a, a)
print(b)"""