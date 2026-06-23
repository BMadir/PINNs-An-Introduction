import numpy as np


class FF_readmesh:
    
    def __init__(self, file):
        
        self.meshfile = np.loadtxt(file, usecols=(0, 1, 2))
        
        self.nv, self.nt, self.ne = self.meshfile[0].astype(int)
        
        self.x = self.vertices[:, 0]
        self.y = self.vertices[:, 1]
        
    @property
    def vertices(self):
        return self.meshfile[1:self.nv+1, :]
    
    @property
    def triangles(self):
        return self.meshfile[self.nv+1:self.nv+self.nt+1, :].astype(int)
    
    @property
    def edges(self):
        return self.meshfile[self.nv+self.nt+1:, :].astype(int)
    
    @property
    def triangulation(self):
        from matplotlib.tri import Triangulation
        
        return Triangulation(x = self.x, y = self.y, triangles = (self.triangles - 1))
        
def FF_readsol(mesh, XYU):
    
    xy  = np.stack([mesh.x, mesh.y]).T
    XY , U = XYU[:, 0:2], XYU[:, -1]
    
    l = []
    for e in xy:
        idx = (XY == e).all(axis=1).nonzero()[0][0]
        l.append(idx)
        
    return l, U[l]
