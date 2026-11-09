# Bratu Adaptive Loss-Weighting Benchmark V2

This benchmark compares adaptive loss-weighting strategies using identical PINN candidate counts and validation-only model selection. The three adaptive methods are LRA, ReLoBRaLo and DDQN, with two static baselines for context.

## Methods

1. **Equal-weight PINN**: \(L=L_f+L_b\).
2. **Fixed-selected PINN**: \(L=L_f+3L_b\), retained as a locked static comparator from the earlier development study. It is not re-tuned per \(\lambda\), and the adaptive-only tables/tests are produced separately so this historical static choice does not define the main contribution.
3. **LRA**: gradient-statistics learning-rate annealing of Wang, Teng and Perdikaris.
4. **ReLoBRaLo**: Relative Loss Balancing with Random Lookback of Bischof and Kraus.
5. **DDQN**: the existing centralized joint controller over the two loss weights.

## Common matched protocol

For every method, \(\lambda\in\{1,2,3,3.4\}\), and outer seed \(s\in\{42,123,456,789,1000\}\):

- five candidate PINNs are trained;
- candidate seed \(j\) is `s + 1000003*j`, for `j=1,...,5`;
- every candidate receives 15,000 PINN optimizer updates;
- architecture, optimizer, collocation grid, validation grid, and test grid are common;
- candidate selection uses only
  \[
  S_{\rm val}=L_{f,{\rm val}}+L_{b,{\rm val}};
  \]
- the numerical reference solution is **not accessed before selection**;
- after the selected candidate is frozen, all five candidates are evaluated against the reference and stored for a post-selection robustness audit.

The post-selection all-candidate metrics do not feed back into training or candidate selection.

## Bratu loss definition

The code uses exactly 256 interior collocation points and exactly 512 interior validation points. Boundary error is evaluated separately as

\[
L_b=\frac{u(0)^2+u(1)^2}{2}.
\]

This matches the mathematical definition in the manuscript rather than the older implementation that summed the two endpoint errors.

## LRA implementation

The implementation in `training/adaptive_weighting.py` intentionally follows the authors' released `GradientPathologiesPINNs` reference code rather than mixing formula variants from later reimplementations.

For the two-term objective

\[
L=L_f+\lambda_b L_b,
\]

the released reference implementation computes the residual maximum gradient statistic and the boundary statistic from the **currently weighted** boundary term. In PyTorch notation the implemented update is

\[
\widehat\lambda_b=
\frac{\max_j\max|\partial L_f/\partial W_j|}
     {\operatorname{mean}_j\operatorname{mean}|\partial(\lambda_b L_b)/\partial W_j|},
\]

followed by

\[
\lambda_b\leftarrow 0.9\lambda_b+0.1\widehat\lambda_b.
\]

Only network weight tensors are used for this statistic, matching the released TensorFlow code's `self.weights`; one-dimensional bias tensors are excluded. The adaptive coefficient is refreshed every 10 optimizer iterations, again matching the released training loop.

Reference implementation:
`PredictiveIntelligenceLab/GradientPathologiesPINNs`.

## ReLoBRaLo implementation

The implementation follows the authors' released `rbischof/relative_balancing` update. For \(m\) loss terms,

\[
\widehat\lambda_i^{(t;t')}=
 m\,\mathrm{softmax}_i\!\left(\frac{L_i(t)}{T L_i(t')}\right),
\]

and the regular update is

\[
\lambda_i(t)=
\rho\alpha\lambda_i(t-1)
+(1-\rho)\alpha\widehat\lambda_i^{(t;0)}
+(1-\alpha)\widehat\lambda_i^{(t;t-1)},
\]

with \(\rho\sim\mathrm{Bernoulli}(p_\rho)\).

The code also reproduces the reference implementation's initialization schedule: effective \(\alpha=1\) on the first update, \(\alpha=0\) on the second update, and the configured \(\alpha\) thereafter. Initial-lookback losses are stored on the second update, as in the released training code.

Final benchmark hyperparameters are fixed before the Bratu runs:

- `alpha = 0.999`
- `P(rho=1) = 0.9999`
- `T = 0.1`

These are literature-derived settings, not values selected using Bratu test errors. ReLoBRaLo uses a private deterministic RNG per candidate so its Bernoulli draws do not perturb the PINN random stream of other methods.

Reference implementation:
`rbischof/relative_balancing`.

## DDQN treatment

DDQN uses the same matched five-candidate budget as the other methods. It receives five PINN candidate trajectories, exactly like the other methods. The controller and replay memory persist across these five trajectories because cross-episode learning is part of the DDQN method. No extra evaluation PINN is trained after the five candidates.

The experiment uses the stated DDQN action, state and reward formulation without further retuning.

## Runtime reporting

Every `.npz` result stores:

- protocol and selection-protocol identifiers;
- all five candidate seeds;
- validation physics/boundary/combined scores;
- training time for each candidate;
- post-selection test `relative_l2`, MSE, MAE and max error for **all five candidates**;
- selected-candidate index and seed;
- method-specific final weights and selected weight trajectory.

The Cluster PBS scripts print host/runtime/package/configuration information at the start of each job.

## Statistical output

`collect_array_results.py` produces both contextual five-method analysis and an adaptive-only analysis for **LRA, ReLoBRaLo, and DDQN**. The inferential tests are symmetric:

- Friedman omnibus test on matched cases;
- all two-sided paired Wilcoxon comparisons;
- Holm multiplicity correction.

There are no one-sided tests that assume DDQN should win.

## Protocol identifiers

- `protocol_version = bratu_adaptive_benchmark_v2`
- `selection_protocol = matched_best_of_n_validation_v2`

The collector rejects results with older protocol identifiers, so legacy outputs can remain on Cluster without silently entering the new tables.
