# Figure reproduction and source data

The plotting command `python scripts/figures/reproduce_all_figures.py` writes `build/figures/Fig1.pdf` through `Fig9.pdf`. The command uses archived numerical data and does not retrain PINNs.

| Figure | Input | Generator |
|---|---|---|
| 1: selected-error median/IQR | `results/final/final_results.csv` | `scripts/figures/generate_article_outputs.py` |
| 2: all-candidate error distributions | `results/final/candidate_results.csv` | `scripts/figures/generate_article_outputs.py` |
| 3: adaptive-method mean ranks | `results/final/final_results.csv` | `scripts/figures/generate_article_outputs.py` |
| 4: selected error vs. elapsed wall-clock | `results/final/final_results.csv` (elapsed-time and selected-error fields) | `scripts/figures/make_time_error_figure.py` |
| 5: pointwise errors | `results/runs/{lra,relobralo,ddqn}_lambda_3p4_seed_123.npz` | `scripts/figures/generate_article_outputs.py` |
| 6: weight trajectories | same three selected-run NPZ files | `scripts/figures/generate_article_outputs.py` |
| 7: Bratu bifurcation | analytical branch formulas | `scripts/figures/make_branch_figures.py` |
| 8: selected solution overlays | archived selected-run profiles at four parameter values | `scripts/figures/make_branch_figures.py` |
| 9: upper/lower branch comparison | selected DDQN and LRA profiles at `lambda=3.4`, outer seed `456` | `scripts/figures/make_branch_figures.py` |

The original general-purpose article plotting script also implements an earlier aggregate-time illustration; the dedicated Figure 4 generator is authoritative for the current accuracy-versus-wall-clock figure.

Under the plotting environment used for checking this package, Figures 1–6 matched the corresponding reference PDF renders. Figures 7–9 were reconstructed from archived profiles and analytical solutions; their contents were validated numerically, but byte-identical page rendering across plotting environments is not guaranteed. For Figure 9, the selected DDQN prediction has RMS distances of approximately `0.060251` to the lower branch and `0.478886` to the upper branch. Differences among Matplotlib and font versions can change typography without changing the underlying data.

Data-integrity hashes for the included reference files are stored in `docs/data_manifest.json`.
