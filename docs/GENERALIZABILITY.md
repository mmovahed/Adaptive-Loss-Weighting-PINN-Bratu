# Complementary PDE experiments

Two supplementary benchmarks test whether the adaptive-method ordering on Bratu persists when the equation or number of loss components changes. Each compares `fixed_equal`, `lra`, `relobralo`, and `ddqn` on five outer seeds, with five matched PINN candidates per method and **15,000 updates** per candidate: **40 five-candidate runs** (200 PINN trainings) in total.

## Two-dimensional Helmholtz

On `(x,y) in (0,1)^2`, with `k=1`:

`Delta u + k^2 u = (k^2 - 2*pi^2) sin(pi*x) sin(pi*y)`.

The exact solution is `sin(pi*x) sin(pi*y)` with homogeneous Dirichlet conditions. The training objective has physics and boundary terms.

## Time-dependent viscous Burgers

On `x in (-1,1)`, `t in (0,1)`, with viscosity `nu=0.05`:

`u_t + u*u_x - nu*u_xx = 0`.

An exact travelling wave is `u(x,t) = c - a*tanh(a*(x-c*t)/(2*nu))` with `c=0.3`, `a=0.6`. Initial and boundary conditions are obtained by evaluating this exact function on the corresponding edges. This objective has distinct physics, initial and boundary terms.

## Matched evaluation and method adaptations

The selected candidate is determined by the sum of unweighted validation losses without using its reference error. LRA updates auxiliary coefficients separately relative to the physics loss for the three-term problem; ReLoBRaLo uses three weights; DDQN expands the joint action space from 25 to 125 actions. These extensions make the comparison feasible but are not literally identical to the two-term Bratu implementations. The five-seed PDE results are descriptive rather than a proof of universal rankings.

The data are in `results/generalizability_v1/production/`. Model definitions are in `problems/generalizability_benchmarks.py`; the run script is `experiments/generalizability_worker.py`; the summary collector is `analysis/collect_generalizability.py`.

## Re-execution

```bash
python -m unittest discover -s tests -p 'test_generalizability.py'
qsub cluster/bratu_generalizability_smoke.pbs
qsub cluster/bratu_generalizability_array.pbs
# After successful completion of the full array:
qsub cluster/bratu_generalizability_collect.pbs
```

The smoke-test records are stored separately from production outputs. Scheduler scripts require a locally configured cluster runtime.
