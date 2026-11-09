# DDQN weight-normalization experiment

This experiment compares the original DDQN weight update with a normalized update under otherwise matched conditions. The original controller independently clips weights into `[0.01,100]`; the normalized variant keeps `alpha + beta = 2` (subject to its positivity safeguard). Both use the same 25 joint multiplicative actions, Q-network architecture, reward, replay protocol, candidate seeds, and 15,000-update PINN training budget.

## Design and outputs

There are 4 Bratu parameters × 5 outer seeds × 2 variants = **40 five-candidate runs** (200 PINN trainings). The original controller and replay memory persist across the five candidates within each run. Every five-candidate run uses **3,700 controller decisions** and **3,669 Q-network optimizer updates**. Timing fields distinguish Q-network updates and validation/controller activity.

Files are stored under `results/ddqn_ablation_v1/production/runs/`, with derived tables under `results/ddqn_ablation_v1/production/summary/`. The collector preserves selected and nonselected candidates, including high-error outcomes.

## Interpretation

The normalized variant improves mean selected error near `lambda=3.4` but not uniformly across Bratu parameters. Normalization reduces the upper tail of all-candidate errors; it does not constitute a controlled test of resetting replay memory or a general ranking of reinforcement-learning-based weighting algorithms. Observed timing differences do not, by themselves, establish processor-independent speedups.

## Re-execution

With the project root and runtime configured for the target cluster:

```bash
python -m unittest discover -s tests -p 'test_ddqn_ablation.py'
qsub cluster/bratu_ddqn_ablation_array.pbs
# After all 40 cases finish:
qsub cluster/bratu_ddqn_ablation_collect.pbs
```

For development-only checks, use the experiment's `--smoke` option. Smoke-test outputs remain separate from `production` records.
