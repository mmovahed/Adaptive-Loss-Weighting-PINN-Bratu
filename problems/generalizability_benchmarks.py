"""Two manufactured/reference-exact PDE benchmarks for PINN loss-weighting comparisons.

Helmholtz: two *spatial* dimensions, two independent objective components.
Burgers:  one spatial + time dimension, THREE independent components.
All training/validation grids are deterministic and disjoint.
"""
import math
import numpy as np
import torch


def _interior_grid(n, a, b, c, d, device):
    n = int(n)
    if n < 1:
        raise ValueError('n must be positive')
    nx = max(1, int(math.floor(math.sqrt(n))))
    ny = int(math.ceil(float(n) / float(nx)))
    xx = torch.linspace(a, b, nx + 2, device=device)[1:-1]
    yy = torch.linspace(c, d, ny + 2, device=device)[1:-1]
    return torch.stack([xx.repeat_interleave(ny), yy.repeat(nx)], 1)[:n]


def _tensor1d(a, b, n, device, interior=False):
    return torch.linspace(a, b, n + 2 if interior else n, device=device)[1:-1] if interior else torch.linspace(a, b, n, device=device)


class Helmholtz2D:
    """Delta(u) + k^2 u = f, u=0 on boundary of [0,1]^2.

    Exact reference u(x,y) = sin(pi*x)sin(pi*y), with
    f(x,y) = (k^2 - 2*pi^2)*u(x,y).
    """
    name = 'helmholtz2d'
    loss_names = ('physics', 'boundary')
    input_dim = 2

    def __init__(self, k=1.0):
        self.k = float(k)

    def training_grids(self, n_collocation, device):
        xy = _interior_grid(n_collocation, 0, 1, 0, 1, device)
        q = _tensor1d(0, 1, 64, device)
        z = torch.zeros_like(q)
        o = torch.ones_like(q)
        boundary = torch.cat([torch.stack([q,z],1), torch.stack([q,o],1),
                              torch.stack([z,q],1), torch.stack([o,q],1)],0)
        return xy, boundary

    def validation_grids(self, n_validation, device):
        # A shifted, fixed interior grid avoids reusing identical points.
        n = int(n_validation)
        nx = max(1, int(math.floor(math.sqrt(n))))
        ny = int(math.ceil(float(n) / nx))
        xx = (torch.arange(nx, device=device).float() + 0.5) / nx
        yy = (torch.arange(ny, device=device).float() + 0.5) / ny
        xy = torch.stack([xx.repeat_interleave(ny),yy.repeat(nx)],1)[:n]
        q = (torch.arange(96,device=device).float() + .5) / 96
        z = torch.zeros_like(q); o = torch.ones_like(q)
        boundary = torch.cat([torch.stack([q,z],1),torch.stack([q,o],1),
                              torch.stack([z,q],1),torch.stack([o,q],1)],0)
        return xy, boundary

    def exact(self, xy):
        return torch.sin(math.pi * xy[:,0:1])*torch.sin(math.pi * xy[:,1:2])

    def physics_residual(self, model, xy):
        xy = xy.detach().clone().requires_grad_(True)
        u = model(xy)
        g = torch.autograd.grad(u,xy,torch.ones_like(u),create_graph=True)[0]
        dxx = torch.autograd.grad(g[:,0:1],xy,torch.ones_like(g[:,0:1]),create_graph=True)[0][:,0:1]
        dyy = torch.autograd.grad(g[:,1:2],xy,torch.ones_like(g[:,1:2]),create_graph=True)[0][:,1:2]
        f = (self.k**2 - 2*math.pi**2)*self.exact(xy)
        return dxx + dyy + self.k**2*u - f

    def loss_components(self, model, grids):
        xy, bc = grids
        lf = torch.mean(self.physics_residual(model,xy)**2)
        lb = torch.mean(model(bc)**2)
        return [lf,lb]

    def test_points(self, device, n_test):
        n = max(4,int(math.sqrt(int(n_test))))
        x = torch.linspace(0,1,n,device=device)
        return torch.stack([x.repeat_interleave(n),x.repeat(n)],1)


class ViscousBurgers1D:
    """Unforced time-dependent viscous Burgers, with an exact traveling wave.

    u_t + u*u_x - nu*u_xx = 0 on [-1,1] x [0,1].
    u(x,t)=c-a*tanh(a*(x-c*t)/(2*nu)).
    Independent equation, initial, and boundary losses.
    """
    name = 'burgers1d_time'
    loss_names = ('physics','initial','boundary')
    input_dim = 2

    def __init__(self, nu=0.05,c=0.3,a=0.6):
        if float(nu)<=0:
            raise ValueError('nu must be positive')
        self.nu=float(nu); self.c=float(c); self.a=float(a)

    def exact(self, xt):
        x=xt[:,0:1]; t=xt[:,1:2]
        return self.c-self.a*torch.tanh(self.a*(x-self.c*t)/(2*self.nu))

    def training_grids(self, n_collocation, device):
        xt=_interior_grid(n_collocation,-1,1,0,1,device)
        x = torch.linspace(-1,1,64+2,device=device)[1:-1]
        initial=torch.stack([x,torch.zeros_like(x)],1)
        t=torch.linspace(0,1,64,device=device)
        boundary=torch.cat([torch.stack([-torch.ones_like(t),t],1),
                            torch.stack([torch.ones_like(t),t],1)],0)
        return xt,initial,boundary

    def validation_grids(self,n_validation,device):
        n=int(n_validation)
        nx=max(1,int(math.sqrt(n))); nt=int(math.ceil(float(n)/nx))
        x=-1+2*(torch.arange(nx,device=device).float()+.5)/nx
        t=(torch.arange(nt,device=device).float()+.5)/nt
        xt=torch.stack([x.repeat_interleave(nt),t.repeat(nx)],1)[:n]
        x0=-1+2*(torch.arange(96,device=device).float()+.5)/96
        initial=torch.stack([x0,torch.zeros_like(x0)],1)
        tb=(torch.arange(96,device=device).float()+.5)/96
        boundary=torch.cat([torch.stack([-torch.ones_like(tb),tb],1),
                            torch.stack([torch.ones_like(tb),tb],1)],0)
        return xt,initial,boundary

    def physics_residual(self,model,xt):
        xt=xt.detach().clone().requires_grad_(True)
        u=model(xt)
        g=torch.autograd.grad(u,xt,torch.ones_like(u),create_graph=True)[0]
        ux=g[:,0:1]; ut=g[:,1:2]
        uxx=torch.autograd.grad(ux,xt,torch.ones_like(ux),create_graph=True)[0][:,0:1]
        return ut + u*ux - self.nu*uxx

    def loss_components(self,model,grids):
        xt,initial,boundary=grids
        physics=torch.mean(self.physics_residual(model,xt)**2)
        initial_loss=torch.mean((model(initial)-self.exact(initial))**2)
        boundary_loss=torch.mean((model(boundary)-self.exact(boundary))**2)
        return [physics,initial_loss,boundary_loss]

    def test_points(self,device,n_test):
        nx=max(4,int(math.sqrt(int(n_test))))
        nt=max(4,int(math.ceil(float(n_test)/nx)))
        x=torch.linspace(-1,1,nx,device=device)
        t=torch.linspace(0,1,nt,device=device)
        return torch.stack([x.repeat_interleave(nt),t.repeat(nx)],1)


def make_problem(name):
    if name=='helmholtz2d': return Helmholtz2D(k=1.0)
    if name=='burgers1d_time': return ViscousBurgers1D(nu=0.05)
    raise ValueError('Unknown benchmark '+str(name))
