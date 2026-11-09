# Validation-only fixed-weight selection: experimental protocol

The historical contextual baseline `fixed_selected=(1,3)` was chosen during reference-error-guided development. This separate experiment evaluates selection of a fixed weight pair using **only unweighted validation losses**, with development seeds disjoint from evaluation seeds.

## Selection stage

The candidate grid is `(1,0.1)`, `(1,0.3)`, `(1,1)`, `(1,3)`, `(1,10)`. The training parameters are `lambda ∈ {1,2,3,3.4}` and development seeds `{2027,2029,2039}`, totaling **60 PINN fits**. Each fit has 15,000 Adam updates and uses the Bratu network, collocation, and validation settings of the main benchmark.

For each configuration, define the unweighted validation objective `S_val = Lf_val + Lb_val`, and aggregate across all 12 development cases using

`J(alpha,beta) = (1/12) * sum_(lambda,s) log(max(S_val(lambda,s;alpha,beta), 1e-30))`.

Select the pair minimizing `J` and lock it for evaluation. The reference solution is not consulted during tuning. The stored `results/validation_static_v1/production/selected_weights.json` selects **(1,3)** from this grid. All 60 trial records are preserved in `results/validation_static_v1/production/tuning/`.

## Evaluation stage

The selected pair is evaluated independently using the five original outer seeds `{42,123,456,789,1000}`, four parameter values, and five candidate PINNs per case, for **100 evaluation fits**. Candidate selection uses unweighted validation loss; reference-solution metrics are calculated only after selection. This is kept separate from the earlier `fixed_selected` result and its development provenance.

The complete stored results are in `results/validation_static_v1/production/evaluation/`; tables derived from them are in `results/validation_static_v1/production/summary/`. Including selection and evaluation, the procedure uses **160 full-budget PINN fits**; the 60-fit tuning cost must not be conflated with the cost of evaluating the locked pair.

## Re-execution

From the repository root, configure the Python runtime and PBS project location as appropriate, then run the stages in order:

```bash
qsub cluster/bratu_static_validation_tune_array.pbs
qsub cluster/bratu_static_validation_finalize.pbs
qsub cluster/bratu_fixed_validation_array.pbs
qsub cluster/bratu_fixed_validation_collect.pbs
```

Submit each subsequent stage only after successful completion of the preceding one. The implementation includes guards against test-reference access during weight selection, output overwrites, and mismatched source fingerprints. For a local protocol test:

```bash
python -m unittest discover -s tests -p 'test_static_validation_protocol.py'
```

The training-time metrics exclude PBS queue waiting time. Historical results should be regenerated only with an appropriately configured runtime, rather than treated as bitwise invariant across Python and PyTorch versions.
