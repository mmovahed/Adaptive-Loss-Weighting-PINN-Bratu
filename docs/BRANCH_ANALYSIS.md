# Bratu solution-branch assessment

For each parameter below the critical value, the analytical Bratu solutions can be expressed as

`u(x;a) = 2 log(cosh(a/2)/cosh(a*(x-1/2)))`, with `lambda = 2*a**2 / cosh(a/2)**2`.

The lower and upper branches can be compared directly against stored selected predictions. A triangle-inequality sufficient condition also identifies many candidate profiles as closer to the lower branch from their archived relative error alone. It establishes lower-branch proximity for **494 of the 500 original candidates**; direct comparison of the 100 archived selected predictions also favors the lower branch.

Six original nonselected DDQN candidates were not classifiable from stored scalar error alone. Five complete five-candidate DDQN runs were replayed because the controller and replay memory are shared across candidates. The replay agreed with the archived error for four of the six; their reconstructed profiles are nearer the lower branch. Two original nonselected candidates did not reproduce their prior errors and remain **unclassified**. A failed replay match is not evidence of convergence to the upper branch.

## Data and scripts

- `reference/branch_separation_bounds.csv`: sufficient lower-branch condition for all parameters.
- `reference/selected_branch_distances.csv`: direct selected-prediction distances.
- `reference/six_unresolved_candidates.csv`: the six cases requiring additional scrutiny.
- `results/branch_audit_v1/production/`: targeted replay NPZ files and comparison tables.
- `experiments/bratu_branch_targeted_replay.py`: replay experiment.
- `analysis/collect_branch_audit.py`: replay comparison/collection.
- `scripts/figures/make_branch_figures.py`: Figures 7–9.

The complete replay can be submitted with `cluster/bratu_branch_audit_array.pbs`, followed by `cluster/bratu_branch_audit_collect.pbs` after successful completion. Paths and runtime must be configured for the local environment.
