#!/usr/bin/env python3
"""Generate publication-ready figures and tables for the Bratu adaptive-weighting benchmark.

Usage:
    python generate_article_outputs.py --results /path/to/results --out /path/to/article_outputs

Expected input layout:
    results/
      final/final_results.csv
      final/candidate_results.csv
      runs/*.npz

The script validates the matched-best-of-five protocol, recomputes the main
statistics, and writes PDF+PNG figures, CSV+LaTeX tables, and a short summary.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

ADAPTIVE = ["lra", "relobralo", "ddqn"]
ALL_METHODS = ["fixed_equal", "fixed_selected", "lra", "relobralo", "ddqn"]
LABELS = {
    "fixed_equal": "Equal-weight",
    "fixed_selected": "Selected fixed",
    "lra": "LRA",
    "relobralo": "ReLoBRaLo",
    "ddqn": "DDQN",
}
MARKERS = {"fixed_equal": "o", "fixed_selected": "s", "lra": "^", "relobralo": "D", "ddqn": "v"}
LINESTYLES = {"fixed_equal": "-", "fixed_selected": "--", "lra": "-", "relobralo": "--", "ddqn": ":"}


def _fmt_sci(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    return rf"${mant:.{digits}f}\times10^{{{exp}}}$"


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _holm_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m, dtype=float)
    running = 0.0
    for j, idx in enumerate(order):
        adjusted = (m - j) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(running, 1.0)
    return out


def validate_inputs(final: pd.DataFrame, cand: pd.DataFrame) -> None:
    required_final = {
        "method", "lambda", "seed", "state", "n_candidates", "selected_candidate",
        "relative_l2", "candidate_train_seconds_total", "protocol_version", "selection_protocol"
    }
    required_cand = {
        "method", "lambda", "outer_seed", "candidate", "candidate_seed", "selected",
        "validation_score", "test_relative_l2", "train_seconds"
    }
    missing_f = required_final - set(final.columns)
    missing_c = required_cand - set(cand.columns)
    if missing_f or missing_c:
        raise ValueError(f"Missing columns. final={missing_f}, candidate={missing_c}")

    if not (final["state"] == "success").all():
        bad = final[final["state"] != "success"]
        raise ValueError(f"Not all final cases succeeded:\n{bad.to_string(index=False)}")

    if final["protocol_version"].nunique() != 1 or final["selection_protocol"].nunique() != 1:
        raise ValueError("Mixed protocol versions detected.")

    expected_methods = set(ALL_METHODS)
    got_methods = set(final["method"].unique())
    if got_methods != expected_methods:
        raise ValueError(f"Expected methods {sorted(expected_methods)}, got {sorted(got_methods)}")

    if len(final) != 100:
        raise ValueError(f"Expected 100 final cases, got {len(final)}")
    if len(cand) != 500:
        raise ValueError(f"Expected 500 candidate trajectories, got {len(cand)}")

    grp = cand.groupby(["method", "lambda", "outer_seed"], sort=False)
    counts = grp.size()
    if not (counts == 5).all():
        raise ValueError("Every method-lambda-seed case must contain exactly five candidates.")
    selected_counts = grp["selected"].sum()
    if not (selected_counts == 1).all():
        raise ValueError("Every method-lambda-seed case must contain exactly one selected candidate.")

    # Cross-check final selected error against candidate table.
    sel = cand[cand["selected"] == 1].rename(columns={"outer_seed": "seed", "test_relative_l2": "cand_rel"})
    merged = final.merge(sel[["method", "lambda", "seed", "candidate", "cand_rel"]],
                         left_on=["method", "lambda", "seed", "selected_candidate"],
                         right_on=["method", "lambda", "seed", "candidate"], how="left")
    if merged["cand_rel"].isna().any():
        raise ValueError("Could not match some selected candidates to final_results.csv")
    if not np.allclose(merged["relative_l2"], merged["cand_rel"], rtol=1e-8, atol=1e-12):
        raise ValueError("Selected candidate errors disagree with final_results.csv")


def summarize_selected(final: pd.DataFrame) -> pd.DataFrame:
    s = (final.groupby(["method", "lambda"], as_index=False)
         .agg(n=("relative_l2", "size"),
              mean=("relative_l2", "mean"),
              std=("relative_l2", "std"),
              median=("relative_l2", "median"),
              q25=("relative_l2", lambda x: x.quantile(0.25)),
              q75=("relative_l2", lambda x: x.quantile(0.75)),
              total_seconds_mean=("candidate_train_seconds_total", "mean"),
              total_seconds_std=("candidate_train_seconds_total", "std")))
    return s


def ranks(final: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = final[final.method.isin(methods)][["method", "lambda", "seed", "relative_l2"]].copy()
    d["rank"] = d.groupby(["lambda", "seed"])["relative_l2"].rank(method="average", ascending=True)
    summary = (d.groupby("method", as_index=False)
               .agg(n_cases=("rank", "size"), mean_rank=("rank", "mean")))
    winners = d.loc[d.groupby(["lambda", "seed"])["relative_l2"].idxmin()].groupby("method").size()
    summary["wins"] = summary["method"].map(winners).fillna(0).astype(int)
    return d, summary


def recompute_stats(final: pd.DataFrame, methods: list[str], scope: str) -> pd.DataFrame:
    pivot = final[final.method.isin(methods)].pivot(index=["lambda", "seed"], columns="method", values="relative_l2")
    pivot = pivot.dropna()
    arrays = [pivot[m].to_numpy() for m in methods]
    stat, p = friedmanchisquare(*arrays)
    rows = [{"scope": scope, "test": "Friedman", "comparison": "all", "n_cases": len(pivot),
             "statistic": stat, "p_value": p, "p_holm": np.nan}]
    pairs = []
    raw_p = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            w = wilcoxon(pivot[a].to_numpy(), pivot[b].to_numpy(), alternative="two-sided")
            pairs.append((a, b, float(w.statistic), float(w.pvalue)))
            raw_p.append(float(w.pvalue))
    adj = _holm_adjust(raw_p)
    for (a, b, st, pv), ph in zip(pairs, adj):
        rows.append({"scope": scope, "test": "Wilcoxon_two_sided", "comparison": f"{a} vs {b}",
                     "n_cases": len(pivot), "statistic": st, "p_value": pv, "p_holm": ph})
    return pd.DataFrame(rows)


def candidate_summary(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, lam), g in cand.groupby(["method", "lambda"], sort=False):
        x = g["test_relative_l2"].to_numpy()
        rows.append({
            "method": method, "lambda": lam, "n_candidates": len(x),
            "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)),
            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75)), "q90": float(np.quantile(x, .90)),
            "max": float(np.max(x)), "frac_gt_1e-2": float(np.mean(x > 1e-2)),
            "seconds_mean": float(g["train_seconds"].mean()),
            "seconds_std": float(g["train_seconds"].std(ddof=1)),
        })
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_selected_accuracy(summary: pd.DataFrame, methods: list[str], out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for m in methods:
        g = summary[summary.method == m].sort_values("lambda")
        ax.errorbar(g["lambda"], g["mean"], yerr=g["std"], marker=MARKERS[m],
                    linestyle=LINESTYLES[m], linewidth=1.6, capsize=3, label=LABELS[m])
    ax.set_yscale("log")
    ax.set_xlabel(r"Bratu parameter $\lambda$")
    ax.set_ylabel(r"Selected relative $L_2$ error (mean $\pm$ SD)")
    ax.set_xticks(sorted(summary["lambda"].unique()))
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    savefig(fig, out / name)


def fig_selected_accuracy_median_iqr(summary: pd.DataFrame, methods: list[str], out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for m in methods:
        g = summary[summary.method == m].sort_values("lambda")
        lower = (g["median"] - g["q25"]).to_numpy()
        upper = (g["q75"] - g["median"]).to_numpy()
        yerr = np.vstack([lower, upper])
        ax.errorbar(g["lambda"], g["median"], yerr=yerr, marker=MARKERS[m],
                    linestyle=LINESTYLES[m], linewidth=1.6, capsize=3, label=LABELS[m])
    ax.set_yscale("log")
    ax.set_xlabel(r"Bratu parameter $\lambda$")
    ax.set_ylabel(r"Selected relative $L_2$ error (median and IQR)")
    ax.set_xticks(sorted(summary["lambda"].unique()))
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    savefig(fig, out / name)


def fig_candidate_distribution(cand: pd.DataFrame, out: Path) -> None:
    methods = ADAPTIVE
    lambdas = sorted(cand["lambda"].unique())
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.22
    offsets = np.linspace(-width, width, len(methods))
    positions_all, data_all, labels_all = [], [], []
    for li, lam in enumerate(lambdas):
        center = li + 1
        for off, m in zip(offsets, methods):
            x = cand[(cand.method == m) & (cand["lambda"] == lam)]["test_relative_l2"].to_numpy()
            positions_all.append(center + off)
            data_all.append(x)
            labels_all.append(m)
    bp = ax.boxplot(data_all, positions=positions_all, widths=0.17, patch_artist=False,
                    showfliers=True, whis=1.5, medianprops={"linewidth": 1.5})
    # Different line styles for boxes by method; no hard-coded colors required.
    for i, box in enumerate(bp["boxes"]):
        m = labels_all[i]
        box.set_linestyle(LINESTYLES[m])
        box.set_linewidth(1.2)
    for i, med in enumerate(bp["medians"]):
        med.set_linewidth(1.5)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(lambdas) + 1), [str(x) for x in lambdas])
    ax.set_xlabel(r"Bratu parameter $\lambda$")
    ax.set_ylabel(r"Candidate relative $L_2$ error (25 runs per method/$\lambda$)")
    ax.grid(True, which="both", axis="y", linewidth=0.45, alpha=0.35)
    # Legend via dummy artists.
    for m in methods:
        ax.plot([], [], linestyle=LINESTYLES[m], marker=MARKERS[m], label=LABELS[m])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    savefig(fig, out / "fig2_candidate_distribution_adaptive")


def fig_mean_rank(rank_summary: pd.DataFrame, out: Path) -> None:
    rs = rank_summary.set_index("method").loc[ADAPTIVE].sort_values("mean_rank")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(rs))
    ax.scatter(rs["mean_rank"], y, s=70, zorder=3)
    for yi, (m, row) in enumerate(rs.iterrows()):
        ax.text(row["mean_rank"] + 0.035, yi, f"wins = {int(row['wins'])}", va="center", fontsize=9)
    ax.set_yticks(y, [LABELS[m] for m in rs.index])
    ax.set_xlim(0.9, 3.15)
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("Mean rank across 20 matched cases (lower is better)")
    ax.grid(True, axis="x", linewidth=0.45, alpha=0.35)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, out / "fig3_adaptive_mean_rank")


def fig_runtime(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for m in ADAPTIVE:
        g = summary[summary.method == m].sort_values("lambda")
        ax.plot(g["lambda"], g["total_seconds_mean"], marker=MARKERS[m],
                linestyle=LINESTYLES[m], linewidth=1.6, label=LABELS[m])
    ax.set_xlabel(r"Bratu parameter $\lambda$")
    ax.set_ylabel("Mean time for five candidate trainings (s)")
    ax.set_xticks(sorted(summary["lambda"].unique()))
    ax.grid(True, linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out / "fig4_adaptive_computational_cost")


def find_representative_seed(final: pd.DataFrame, lam: float) -> int:
    # Objective choice: seed with median selected LRA error at the hardest lambda.
    g = final[(final.method == "lra") & (final["lambda"] == lam)].sort_values("relative_l2")
    return int(g.iloc[len(g) // 2]["seed"])


def load_npz(run_dir: Path, method: str, lam: float, seed: int):
    lam_token = str(lam).replace(".", "p")
    f = run_dir / f"{method}_lambda_{lam_token}_seed_{seed}.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    return np.load(f, allow_pickle=True)


def fig_pointwise_error(run_dir: Path, final: pd.DataFrame, out: Path) -> tuple[float, int]:
    lam = float(final["lambda"].max())
    seed = find_representative_seed(final, lam)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for m in ADAPTIVE:
        z = load_npz(run_dir, m, lam, seed)
        x = np.asarray(z["x"]).ravel()
        err = np.abs(np.asarray(z["y_pred"]).ravel() - np.asarray(z["y_ref"]).ravel())
        err = np.maximum(err, 1e-16)
        ax.plot(x, err, linestyle=LINESTYLES[m], linewidth=1.5, label=LABELS[m])
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Absolute pointwise error")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out / f"fig5_pointwise_error_lambda{str(lam).replace('.', 'p')}_seed{seed}")
    return lam, seed


def fig_weight_trajectories(run_dir: Path, lam: float, seed: int, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.1), sharex=False)
    for ax, m in zip(axes, ADAPTIVE):
        z = load_npz(run_dir, m, lam, seed)
        step = np.asarray(z["step"]).ravel()
        alpha = np.asarray(z["alpha"]).ravel()
        beta = np.asarray(z["beta"]).ravel()
        ax.plot(step, alpha, linewidth=1.2, label=r"$\alpha$")
        ax.plot(step, beta, linewidth=1.2, linestyle="--", label=r"$\beta$")
        # log scale only when dynamic range warrants it.
        positive = np.concatenate([alpha[alpha > 0], beta[beta > 0]])
        if len(positive) and positive.max() / positive.min() > 50:
            ax.set_yscale("log")
        ax.set_ylabel("Loss weight")
        ax.set_title(LABELS[m], loc="left", fontsize=10)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
        ax.legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Recorded training step")
    fig.tight_layout()
    savefig(fig, out / f"fig6_weight_trajectories_lambda{str(lam).replace('.', 'p')}_seed{seed}")


def write_table1(summary: pd.DataFrame, out: Path) -> None:
    s = summary[summary.method.isin(ADAPTIVE)].copy()
    rows = []
    for lam in sorted(s["lambda"].unique()):
        row = {"lambda": lam}
        for m in ADAPTIVE:
            q = s[(s.method == m) & (s["lambda"] == lam)].iloc[0]
            row[f"{m}_mean"] = q["mean"]
            row[f"{m}_std"] = q["std"]
            row[f"{m}_median"] = q["median"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table1_adaptive_selected_accuracy.csv", index=False)
    lines = [r"\begin{tabular}{c c c c}", r"\toprule",
             r"$\lambda$ & LRA & ReLoBRaLo & DDQN \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        means = {m: r[f"{m}_mean"] for m in ADAPTIVE}
        best = min(means, key=means.get)
        for m in ADAPTIVE:
            txt = rf"{_fmt_sci(r[f'{m}_mean'])} $\pm$ {_fmt_sci(r[f'{m}_std'])}"
            if m == best:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(f"{r['lambda']:.1f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_adaptive_selected_accuracy.tex").write_text("\n".join(lines), encoding="utf-8")


def write_table2(rank_summary: pd.DataFrame, cand: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    rows = []
    costs = summary[summary.method.isin(ADAPTIVE)].groupby("method")["total_seconds_mean"].mean()
    rs_idx = rank_summary.set_index("method")
    for m in ADAPTIVE:
        x = cand[cand.method == m]["test_relative_l2"].to_numpy()
        rows.append({
            "method": m,
            "mean_rank": float(rs_idx.loc[m, "mean_rank"]),
            "wins": int(rs_idx.loc[m, "wins"]),
            "candidate_median": float(np.median(x)),
            "candidate_q90": float(np.quantile(x, .90)),
            "fraction_above_1e-2": float(np.mean(x > 1e-2)),
            "five_candidate_time_s": float(costs[m]),
        })
    rs = pd.DataFrame(rows).sort_values("mean_rank")
    rs.to_csv(out / "table2_adaptive_rank_robustness_cost.csv", index=False)
    lines = [r"\begin{tabular}{l r r r r r}", r"\toprule",
             r"Method & Mean rank & Wins & Candidate median & Candidate Q90 & Five-candidate time (s) \\", r"\midrule"]
    for _, row in rs.iterrows():
        line = (f"{LABELS[row['method']]} & {row['mean_rank']:.2f} & {int(row['wins'])} & "
                f"{_fmt_sci(row['candidate_median'])} & {_fmt_sci(row['candidate_q90'])} & "
                f"{row['five_candidate_time_s']:.1f}" + " " + chr(92) * 2)
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table2_adaptive_rank_robustness_cost.tex").write_text("\n".join(lines), encoding="utf-8")

def write_table3(stats: pd.DataFrame, out: Path) -> None:
    st = stats[stats.scope == "adaptive_methods"].copy()
    st.to_csv(out / "table3_adaptive_statistical_tests.csv", index=False)
    lines = [r"\begin{tabular}{l r r r}", r"\toprule",
             r"Comparison & Statistic & Raw $p$ & Holm-adjusted $p$ \\", r"\midrule"]
    for _, row in st.iterrows():
        if row["test"] == "Friedman":
            comp = "Friedman (all three)"
        else:
            a, b = row["comparison"].split(" vs ")
            comp = f"{LABELS[a]} vs {LABELS[b]}"
        ph = "--" if pd.isna(row["p_holm"]) else f"{row['p_holm']:.4g}"
        lines.append(f"{comp} & {row['statistic']:.3f} & {row['p_value']:.4g} & {ph}" + " " + chr(92) * 2)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table3_adaptive_statistical_tests.tex").write_text("\n".join(lines), encoding="utf-8")

def write_supplement(summary: pd.DataFrame, rank_all: pd.DataFrame, cand: pd.DataFrame, out: Path) -> None:
    # Full five-method accuracy table.
    rows = []
    for lam in sorted(summary["lambda"].unique()):
        for m in ALL_METHODS:
            r = summary[(summary.method == m) & (summary["lambda"] == lam)].iloc[0]
            rows.append({"lambda": lam, "method": m, "mean": r["mean"], "std": r["std"], "median": r["median"]})
    pd.DataFrame(rows).to_csv(out / "tableS1_all_methods_selected_accuracy.csv", index=False)
    rank_all.to_csv(out / "tableS2_all_methods_rank_summary.csv", index=False)

    # Candidate robustness aggregated exactly from raw candidate trajectories.
    rows = []
    for m in ADAPTIVE:
        x = cand[cand.method == m]["test_relative_l2"].to_numpy()
        rows.append({"method": m, "n_candidates": len(x), "mean": x.mean(), "median": np.median(x),
                     "q75": np.quantile(x, .75), "q90": np.quantile(x, .90), "q95": np.quantile(x, .95),
                     "max": x.max(), "fraction_above_1e-2": np.mean(x > 1e-2),
                     "fraction_above_1e-1": np.mean(x > 1e-1)})
    pd.DataFrame(rows).to_csv(out / "tableS3_adaptive_candidate_robustness.csv", index=False)


def write_summary(final: pd.DataFrame, cand: pd.DataFrame, sel_summary: pd.DataFrame,
                  rank_ad: pd.DataFrame, rank_all: pd.DataFrame, stats: pd.DataFrame,
                  rep_lam: float, rep_seed: int, out: Path) -> None:
    ad_stats = stats[stats.scope == "adaptive_methods"]
    fried = ad_stats[ad_stats.test == "Friedman"].iloc[0]
    rs = rank_ad.set_index("method")
    costs = sel_summary[sel_summary.method.isin(ADAPTIVE)].groupby("method")["total_seconds_mean"].mean()

    lines = [
        "# Bratu adaptive-weighting benchmark: generated article outputs",
        "",
        f"Protocol: `{final['protocol_version'].iloc[0]}` / `{final['selection_protocol'].iloc[0]}`.",
        f"Validated {len(final)} successful selected cases and {len(cand)} candidate trajectories.",
        "",
        "## Main numerical story",
        "",
        f"- Adaptive-only Friedman test across 20 matched cases: statistic = {fried['statistic']:.3f}, p = {fried['p_value']:.6g}.",
        f"- Mean ranks: LRA {rs.loc['lra','mean_rank']:.2f} ({int(rs.loc['lra','wins'])} wins), "
        f"DDQN {rs.loc['ddqn','mean_rank']:.2f} ({int(rs.loc['ddqn','wins'])} wins), "
        f"ReLoBRaLo {rs.loc['relobralo','mean_rank']:.2f} ({int(rs.loc['relobralo','wins'])} wins).",
    ]
    for comp in ["lra vs relobralo", "lra vs ddqn", "relobralo vs ddqn"]:
        r = ad_stats[(ad_stats.test == "Wilcoxon_two_sided") & (ad_stats.comparison == comp)].iloc[0]
        lines.append(f"- {LABELS[comp.split(' vs ')[0]]} vs {LABELS[comp.split(' vs ')[1]]}: raw p={r['p_value']:.6g}, Holm p={r['p_holm']:.6g}.")
    lines += [
        f"- Mean five-candidate training time across lambda: LRA {costs['lra']:.1f}s, ReLoBRaLo {costs['relobralo']:.1f}s, DDQN {costs['ddqn']:.1f}s.",
        "- Best-of-five opportunities, candidate initializations, and validation-based selection are matched across all methods; wall-clock time is not forced to be equal.",
        f"- Figure 5/6 use lambda={rep_lam} and outer seed={rep_seed}, chosen objectively as the median-LRA-error seed at the hardest lambda.",
        "",
        "## Suggested placement",
        "",
        "Main text: Fig. 1 (selected adaptive median/IQR accuracy), Fig. 2 (all-candidate robustness), Fig. 3 (mean ranks), Table 1 (selected mean+-SD accuracy), Table 3 (statistics).",
        "Discussion / secondary main text: Fig. 4 (cost).",
        "Supplement or mechanism analysis: Fig. 5 (pointwise error), Fig. 6 (weight trajectories), Tables S1-S3.",
        "",
        "## Reproduction",
        "",
        "Run:",
        "```bash",
        "python generate_article_outputs.py --results /path/to/results --out /path/to/article_outputs",
        "```",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_captions(rep_lam: float, rep_seed: int, out: Path) -> None:
    text = f"""# Proposed figure captions

**Figure 1.** Selected relative $L_2$ error for the three adaptive loss-weighting methods across the Bratu parameter range. Each point shows the median over five matched outer seeds after the same five-candidate validation-based selection protocol; error bars denote the interquartile range. The logarithmic vertical scale emphasizes regime-dependent changes in accuracy.

**Figure 2.** Distribution of test relative $L_2$ errors over all 25 candidate PINN trajectories per method and $\\lambda$ (five outer seeds times five matched candidate initializations). This view removes dependence on the final best-of-five selection and exposes run-to-run robustness and heavy-tailed failures.

**Figure 3.** Mean rank of the adaptive methods over 20 matched $(\\lambda,\\mathrm{{seed}})$ cases. Lower rank is better; annotations report the number of case-wise wins.

**Figure 4.** Mean cumulative training time required for the five candidate PINNs in each selected case. The experimental protocol matches the number of candidate trainings rather than exact wall-clock time.

**Figure 5.** Pointwise absolute error at $\\lambda={rep_lam}$ and outer seed {rep_seed}. The seed is selected objectively as the median-LRA-error seed among the five outer seeds at the hardest Bratu parameter.

**Figure 6.** Evolution of the two PINN loss weights for the selected trajectories at $\\lambda={rep_lam}$ and outer seed {rep_seed}. The panels illustrate the distinct adaptation mechanisms of LRA, ReLoBRaLo, and DDQN under the same benchmark case.
"""
    (out / "captions.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Root results directory containing final/ and runs/")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    root = Path(args.results).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    final_dir = root / "final"
    run_dir = root / "runs"

    final = pd.read_csv(final_dir / "final_results.csv")
    cand = pd.read_csv(final_dir / "candidate_results.csv")
    validate_inputs(final, cand)

    summary = summarize_selected(final)
    cand_sum = candidate_summary(cand)
    rank_cases_ad, rank_ad = ranks(final, ADAPTIVE)
    rank_cases_all, rank_all = ranks(final, ALL_METHODS)
    stats_ad = recompute_stats(final, ADAPTIVE, "adaptive_methods")
    stats_all = recompute_stats(final, ALL_METHODS, "all_methods")
    stats = pd.concat([stats_ad, stats_all], ignore_index=True)

    # Machine-readable regenerated summaries.
    summary.to_csv(out / "regenerated_selected_summary.csv", index=False)
    cand_sum.to_csv(out / "regenerated_candidate_summary.csv", index=False)
    rank_cases_ad.to_csv(out / "regenerated_adaptive_case_ranks.csv", index=False)
    rank_ad.to_csv(out / "regenerated_adaptive_rank_summary.csv", index=False)
    stats.to_csv(out / "regenerated_statistical_tests.csv", index=False)

    # Figures.
    fig_selected_accuracy_median_iqr(summary, ADAPTIVE, out, "fig1_selected_accuracy_adaptive_median_iqr")
    fig_selected_accuracy(summary, ADAPTIVE, out, "figS2_selected_accuracy_adaptive_mean_sd")
    fig_candidate_distribution(cand[cand.method.isin(ADAPTIVE)], out)
    fig_mean_rank(rank_ad, out)
    fig_runtime(summary, out)
    rep_lam, rep_seed = fig_pointwise_error(run_dir, final, out)
    fig_weight_trajectories(run_dir, rep_lam, rep_seed, out)
    fig_selected_accuracy(summary, ALL_METHODS, out, "figS1_selected_accuracy_all_methods")

    # Tables.
    write_table1(summary, out)
    write_table2(rank_ad, cand, summary, out)
    write_table3(stats, out)
    write_supplement(summary, rank_all, cand, out)
    write_summary(final, cand, summary, rank_ad, rank_all, stats, rep_lam, rep_seed, out)
    write_captions(rep_lam, rep_seed, out)

    print(f"Generated outputs in {out}")
    print(f"Representative hardest-case seed: lambda={rep_lam}, seed={rep_seed}")


if __name__ == "__main__":
    main()
