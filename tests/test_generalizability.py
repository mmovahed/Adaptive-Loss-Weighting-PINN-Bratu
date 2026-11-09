import math
import numpy as np
import torch
from problems.generalizability_benchmarks import Helmholtz2D,ViscousBurgers1D
from experiments.generalizability_worker import index_case,_apply_action,_state,_lra_update_multi


def verify_equations():
    class Exact(torch.nn.Module):
        def __init__(self,p): super().__init__(); self.p=p
        def forward(self,xy): return self.p.exact(xy)
    for p in (Helmholtz2D(),ViscousBurgers1D()):
        pts=p.training_grids(100,'cpu')[0]
        residual=p.physics_residual(Exact(p),pts).detach().numpy()
        assert np.max(abs(residual))<1e-4,(p.name,float(np.max(abs(residual))))
        grids=p.training_grids(100,'cpu')
        terms=[v.item() for v in p.loss_components(Exact(p),grids)]
        assert max(abs(x) for x in terms)<1e-8,(p.name,terms)
        assert len(terms)==len(p.loss_names)
    assert index_case(0)==('helmholtz2d','fixed_equal',42)
    assert index_case(39)==('burgers1d_time','ddqn',1000)
    assert np.allclose(_apply_action(np.ones(3),62),np.ones(3))
    assert _state(np.ones(3),np.ones(3),np.ones(3),1,10).size==10
    assert _state(np.ones(2),np.ones(2),np.ones(2),1,10).size==7
    print('PDE residual/BC/IC exact tests and action/state tests passed.')


if __name__=='__main__': verify_equations()
