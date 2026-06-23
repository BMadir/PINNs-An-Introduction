"""
Gauss-Legendre numerical quadrature utilities.

This module provides:
- 1D and 2D Gauss-Legendre quadrature rules
- Interval scaling
- Numerical integration
"""

import numpy as np
import scipy

class GaussLegendre:

    def __init__(self):
        pass
        
    def scale(self, x, a, b):
        A = (b - a)/2
        B = (b + a)/2
        return A* x + B, A

    def Quad1d(self, ordre):
        x, w = scipy.special.roots_legendre(ordre)
        return x.reshape(-1, 1), w.reshape(-1, 1)
        
    def int1d(self, f, a, b, ordre):
        x, w = self.Quad1d(ordre)
        x, dx = self.scale(x, a, b)
        return (w* f(x)* dx).sum()
        
    def Quad2d(self, ordre):
        l, w = self.Quad1d(ordre)
        x, y = np.meshgrid(l, l)
        return (x, y), w @ w.T
        
    def int2d(self, f, x_b, y_b, ordre):
        (x, y), W = self.Quad2d(ordre)
        x, dx = self.scale(x, x_b[0], x_b[1])
        y, dy = self.scale(y, y_b[0], y_b[1])
        F = f(x, y)* dx* dy
        return np.einsum('ji, ij->j', W, F).sum()
