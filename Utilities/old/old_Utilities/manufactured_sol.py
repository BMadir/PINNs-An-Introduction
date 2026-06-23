import numpy as np

class Nourgaliev:

    def __init__(self, Re=100, Pr=0.71, Ra=1000, K=1, gamma_1=0.1, gamma_2=0.1, P_bar=0, T_bar=1, dP_0=0.1, dU_0=1.0, dT_0=1.0, a_p=0.05, a_u=0.4, a_T=0.1, WithTemp=False):
        self.Re = Re
        self.Ra = Ra
        self.Pr = Pr
        self.K = K
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.P_bar = P_bar
        self.T_bar = T_bar
        self.dP_0 = dP_0
        self.dU_0 = dU_0
        self.dT_0 = dT_0
        self.a_p = a_p
        self.a_u = a_u
        self.a_T = a_T
        self.WithTemp = WithTemp

    def f_B(self, T):
        return T* self.Ra/(self.Pr* self.Re**2)

    def alpha(self, x, t):
        return x + self.gamma_1* t
    def beta(self, y, t):
        return y + self.gamma_2* t

    def u1(self, x, y, t, lib=np):
        alpha = self.alpha(x, t)
        beta = self.beta(y, t)
        u1 = (self.dU_0 + self.a_u * lib.sin(t)) * lib.cos(alpha) * lib.sin(beta)
        return u1

    def u2(self, x, y, t, lib=np):
        alpha = self.alpha(x, t)
        beta = self.beta(y, t)
        u2 = -(self.dU_0 + self.a_u * lib.sin(t)) * lib.sin(alpha) * lib.cos(beta)
        return u2

    def p(self, x, y, t, lib=np):
        alpha = self.alpha(x, t)
        beta = self.beta(y, t)
        return self.P_bar + (self.dP_0 + self.a_p * lib.sin(t)) * lib.sin(alpha) * lib.cos(beta)

    def T(self, x, y, t, lib=np):
        alpha = self.alpha(x, t)
        beta = self.beta(y, t)
        return self.T_bar + (self.dT_0 + self.a_T * lib.sin(t)) * lib.cos(alpha) * lib.sin(beta)

    def grad_p(self, x, y, t, lib=np):
        alpha = self.alpha(x, t)
        beta = self.beta(y, t)
        p_x = (self.dP_0 + self.a_p * lib.sin(t)) * lib.cos(alpha) * lib.cos(beta)
        p_y = -(self.dP_0 + self.a_p * lib.sin(t)) * lib.sin(alpha) * lib.sin(beta)
        return [p_x, p_y]

    def forcing(self, x, y, t, lib=np):
        alpha = self.alpha(x, t)
        beta = self.beta(y, t)
        psi_u = self.dU_0 + self.a_u * lib.sin(t)
        psi_p = self.dP_0 + self.a_p * lib.sin(t)
        psi_T = self.dT_0 + self.a_T * lib.sin(t)

        u1 = self.u1(x, y, t, lib=lib)
        u2 = self.u2(x, y, t, lib=lib)
        T = self.T(x, y, t, lib=lib)

        f1 =self.a_u * lib.cos(t) * lib.cos(alpha) * lib.sin(beta) -\
            self.gamma_1* psi_u* lib.sin(alpha) * lib.sin(beta) +\
            self.gamma_2* psi_u* lib.cos(alpha) * lib.cos(beta) -\
            psi_u* lib.sin(alpha) * lib.sin(beta)* u1 +\
            psi_u* lib.cos(alpha) * lib.cos(beta) * u2 +\
            psi_p* lib.cos(alpha) * lib.cos(beta) + (2/self.Re)* u1

        f2 = -self.a_u * lib.cos(t) * lib.sin(alpha) * lib.cos(beta) -\
             self.gamma_1 * psi_u * lib.cos(alpha) * lib.cos(beta) +\
             self.gamma_2 * psi_u * lib.sin(alpha) * lib.sin(beta) -\
             psi_u * lib.cos(alpha) * lib.cos(beta) * u1 +\
             psi_u * lib.sin(alpha) * lib.sin(beta) * u2 -\
             psi_p * lib.sin(alpha) * lib.sin(beta) + (2 / self.Re) * u2 - self.f_B(T)* self.WithTemp

        f3 = self.a_T * lib.cos(t) * lib.cos(alpha) * lib.sin(beta) -\
             self.gamma_1 * psi_T * lib.sin(alpha) * lib.sin(beta) +\
             self.gamma_2 * psi_T * lib.cos(alpha) * lib.cos(beta) -\
             psi_T * lib.sin(alpha) * lib.sin(beta) * u1 +\
             psi_T * lib.cos(alpha) * lib.cos(beta) * u2 + \
             2* self.K* (1/(self.Re* self.Pr))* psi_T* lib.cos(alpha)* lib.sin(beta)

        if self.WithTemp:
            return [f1, f2, f3]
        else:
            return [f1, f2]