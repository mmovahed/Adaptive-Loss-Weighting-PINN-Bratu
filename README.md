# Reproducibility resources: Adaptive PINN loss weighting

This repository accompanies **Adaptive Loss Weighting in Physics-Informed Neural Networks for the Bratu Problem: A Comparative Study**. It provides the experiment implementations, run-level results (including nonselected and high-error candidates), plotting code, numerical summaries, and consistency tests.

## Repository layout

| Directory | Contents |
|---|---|
| `experiments/`, `training/`, `models/`, `problems/`, `rl/`, `utils/`, `configs/` | PINN training, loss weighting, benchmarks, and experiment configurations |
| `results/final/`, `results/runs/` | Bratu benchmark summaries and individual run records |
| `results/validation_static_v1/` | Validation-only static-weight search and evaluation |
| `results/ddqn_ablation_v1/` | Original-control and normalized-DDQN comparison |
| `results/branch_audit_v1/`, `reference/` | Branch-distance calculations and targeted replay data |
| `results/generalizability_v1/` | Two-dimensional Helmholtz and time-dependent Burgers runs |
| `scripts/figures/`, `scripts/tables/`, `analysis/statistics/` | Figure, table, and statistical reproduction |
| `tests/` | Numerical and experiment-protocol checks |
| `cluster/` | Optional PBS job scripts; paths and Python environments require local configuration |
| `docs/` | Reproduction instructions and figure provenance |

## Reproduce figures and tables (without retraining)

With a Python environment satisfying `requirements-figures.txt`:

```bash
python -m pip install -r requirements-figures.txt
python scripts/figures/reproduce_all_figures.py
python scripts/tables/reproduce_revision_tables.py
python analysis/statistics/verify_statistical_sensitivity.py
python -m unittest discover -s tests -p 'test_figure_repro.py'
```

Figures are written to `build/figures/Fig1.pdf` through `Fig9.pdf`; numerical summary tables are written to `build/tables/`. Generated files under `build/` are not included as research inputs. See [figure reproduction details](docs/FIGURE_PROVENANCE.md).

The figures use archived results and predictions, so the commands above **do not retrain** the PINNs. Figures 1–6 were verified against their corresponding publication figures under the tested plotting environment. Figures 7–9 are generated from archived predictions and analytical Bratu branches; their numerical contents were checked independently, although rendering may vary by Matplotlib version and fonts.

## Experiments

The original Bratu study compares `lra`, `relobralo`, and `ddqn`, with `fixed_equal` and `fixed_selected` as contextual baselines. The benchmark uses four Bratu parameter values (`1`, `2`, `3`, `3.4`), five outer seeds (`42`, `123`, `456`, `789`, `1000`), five matched candidate PINNs per case, and 15,000 optimizer updates per candidate. Candidate selection uses the unweighted validation residual and boundary loss, without accessing reference-solution errors before selection. All candidate test errors are recorded after selection.

Detailed procedures are provided in:

- [Matched Bratu protocol](ADAPTIVE_BENCHMARK_PROTOCOL.md)
- [Validation-only static-weight selection](docs/VALIDATION_STATIC.md)
- [DDQN weight-normalization experiment](docs/DDQN_NORMALIZATION.md)
- [Bratu branch assessment](docs/BRANCH_ANALYSIS.md)
- [Complementary PDE benchmarks](docs/GENERALIZABILITY.md)

## Software and reproducibility scope

The archived CPU experiments used Python 3.7.9 and PyTorch 1.5.1, with single-threaded model training. The analysis and visualization commands use the separate dependencies in `requirements-figures.txt`. Training results may vary across numerical library versions and CPU implementations; byte-for-byte retraining on current environments is not asserted. PBS files are optional orchestration examples and should be configured for the target system.

The dataset contains high-error and nonselected candidates rather than filtering to successful selected predictions. Two original nonselected DDQN candidate profiles cannot be assigned reliably to a Bratu solution branch from the available archived information; their recorded errors remain available, and the branch-analysis documentation explains the limitation.


## Original method references

The Bratu benchmark implements the published LRA and ReLoBRaLo loss-weighting
methods in PyTorch and uses the corresponding research implementations as
algorithmic references. Their upstream code is not included in this repository.

- S. Wang, Y. Teng, and P. Perdikaris, *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*, SIAM Journal on Scientific Computing 43(5), A3055–A3081 (2021). [Reference implementation](https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs).
- R. Bischof and M. Kraus, *Multi-Objective Loss Balancing for Physics-Informed Deep Learning*. [Reference implementation](https://github.com/rbischof/relative_balancing).

## License

The original Python code and associated software documentation are released
under the [MIT License](LICENSE). The original numerical experiment data are
released under [CC BY 4.0](DATA_LICENSE.md). The referenced methods and any
third-party software retain their own intellectual-property rights; this
repository does not include the third-party reference source code.
