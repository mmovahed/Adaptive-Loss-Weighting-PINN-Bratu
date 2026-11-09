CLUSTER ADAPTIVE LOSS-WEIGHTING BENCHMARK V2
===========================================

Expected project path:
  ${BRATU_PROJECT_ROOT}

Main submission:
  cd ${BRATU_PROJECT_ROOT}
  qsub cluster/bratu_array.pbs

Array design:
  5 methods x 4 lambdas x 5 outer seeds = 100 jobs

Methods:
  fixed_equal     : alpha=1, beta=1
  fixed_selected  : locked static comparator alpha=1, beta=3
  lra             : Wang-Teng-Perdikaris reference-code implementation
  relobralo       : Bischof-Kraus reference-code update and bootstrap
  ddqn            : joint DDQN adaptive controller

Lambdas:
  1.0, 2.0, 3.0, 3.4

Outer seeds:
  42, 123, 456, 789, 1000

PROTOCOL TAGS
-------------
  protocol          = bratu_adaptive_benchmark_v2
  selection_protocol= matched_best_of_n_validation_v2

FAIR CANDIDATE PROTOCOL
-----------------------
For every (method, lambda, outer_seed):
  - train 5 candidate PINNs;
  - use the same deterministic candidate seeds for every method;
  - each candidate receives 15000 PINN optimizer updates;
  - select by unweighted S_val = Lf_val + Lb_val only;
  - DO NOT access the numerical reference before selection;
  - after selection is frozen, evaluate ALL five candidates against the
    reference and save those metrics as a post-selection robustness audit.

Candidate seeds are:
  outer_seed + 1000003 * {1,2,3,4,5}

This closes the old best-of-five asymmetry: every method receives the same
number of PINN trajectories, the same candidate initializations and the same
validation-only selection rule.

METHOD-SPECIFIC SETTINGS
------------------------
LRA:
  smoothing/old-weight retention = 0.9
  coefficient update interval    = 10 optimizer iterations
  implementation matches the released GradientPathologiesPINNs reference code:
    - gradient statistics use network weight tensors (not bias tensors);
    - numerator is maximum absolute residual-loss gradient;
    - denominator is the mean of layer-wise mean absolute gradients of the
      CURRENTLY WEIGHTED boundary loss.

ReLoBRaLo:
  alpha                = 0.999
  P(rho=1)             = 0.9999
  temperature T        = 0.1
  implementation follows the authors' released relative_balancing update,
  including its alpha bootstrap schedule (1, 0, then configured alpha) and
  Bernoulli random lookback. A private per-candidate RNG is used so its draws do
  not perturb PINN initialization/randomness of other methods.

DDQN:
  candidate episodes   = 5
  warm-up              = 200 PINN updates
  decision interval    = 20 PINN updates
  joint actions        = 25 multiplier pairs from {0.8,0.9,1,1.1,1.2}^2
  controller/replay memory persist across the five candidate PINNs;
  there is NO fresh sixth evaluation PINN in this benchmark protocol.

COMMON NUMERICAL SETTINGS
-------------------------
  PINN                  : 1-64-64-64-1, tanh
  Adam learning rate    : 1e-3
  interior collocation  : exactly 256 points
  interior validation   : exactly 512 points
  test                  : 2000 points including endpoints
  boundary loss         : mean[u(0)^2, u(1)^2]

CLUSTER CONFIGURATION LOGGING
----------------------------
The PBS script prints the cluster/runtime configuration before Python starts.
Python then prints the effective package and experiment configuration again.
Per-case Python output is streamed with tee into:
  results/array_logs/
Status files are written to:
  results/array_status/
Numerical outputs are written to:
  results/runs/

Python selection:
  primary : ${BRATU_WORK_BASE}/runtime/python37_glibc217
  fallback: ${BRATU_PYTHON_ENV}/bin/python
The selected path is printed at the beginning of every job.

PROGRESS
--------
  qstat
  bash cluster/check_progress.sh

The progress script counts ONLY protocol-v2 status files. Old runs cannot make
v2 progress appear complete.

COLLECTION
----------
After the 100 jobs finish:
  qsub cluster/collect_results.pbs

The collector rejects stale protocols and creates:
  results/final/final_results.csv
  results/final/method_summary.csv
  results/final/candidate_results.csv
  results/final/candidate_method_summary.csv
  results/final/case_ranks.csv
  results/final/method_rank_summary.csv
  results/final/adaptive_case_ranks.csv
  results/final/adaptive_method_rank_summary.csv
  results/final/statistical_tests.csv
  results/final/failed_or_missing_cases.txt

candidate_results.csv contains 500 rows when all runs succeed: every candidate's
validation score, post-selection test metrics, final weights, runtime and whether
it was selected. This makes it possible to report both selected-model performance
and all-candidate distributions without discarding the non-selected trajectories.

statistical_tests.csv uses matched two-sided inference. It reports Friedman tests
and two-sided paired Wilcoxon tests with Holm correction for:
  (a) all five methods, and
  (b) the three adaptive methods LRA, ReLoBRaLo and DDQN.
No method receives a one-sided test advantage.

RERUNS
------
Example failed case:
  qsub -t 37 cluster/bratu_array.pbs

DDQN-only IDs 80..99 can be rerun with the backward-compatible script name:
  qsub cluster/bratu_ddqn_online_array.pbs
Despite the historical filename, that script now runs the benchmark-v2 DDQN
protocol, not the old fresh-PINN online-evaluation protocol.

IMPORTANT ABOUT OLD RESULTS
---------------------------
Old fixed_tuned/single_agent/pre-v2 .npz, logs and statuses may remain on disk.
They are intentionally NOT deleted. The collector requires both protocol-v2 tags
and the new result filenames (fixed_selected, ddqn), so old outputs cannot enter
new tables silently.
