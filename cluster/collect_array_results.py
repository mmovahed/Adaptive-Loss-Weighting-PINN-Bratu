#!/usr/bin/env python
from __future__ import print_function

import csv
import itertools
import math
from pathlib import Path

import numpy as np

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except Exception:
    friedmanchisquare = None
    wilcoxon = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "results" / "runs"
STATUS_DIR = PROJECT_ROOT / "results" / "array_status"
FINAL_DIR = PROJECT_ROOT / "results" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PROTOCOL = "bratu_adaptive_benchmark_v2"
EXPECTED_SELECTION_PROTOCOL = "matched_best_of_n_validation_v2"
METHODS = ["fixed_equal", "fixed_selected", "lra", "relobralo", "ddqn"]
ADAPTIVE_METHODS = ["lra", "relobralo", "ddqn"]
LAMBDAS = [1.0, 2.0, 3.0, 3.4]
SEEDS = [42, 123, 456, 789, 1000]
EXPECTED_CANDIDATES = 5


def lam_tag(lam):
    return str(lam).replace(".", "p")


def parse_status(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def scalar_text(data, key, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def scalar_float(data, key, default=None):
    if key not in data.files:
        return default
    try:
        return float(np.asarray(data[key]).reshape(-1)[0])
    except Exception:
        return default


def array_float(data, key, n=None, fill=None):
    if key in data.files:
        arr = np.asarray(data[key]).reshape(-1).astype(float)
    elif n is not None:
        arr = np.asarray([fill] * n, dtype=float)
    else:
        arr = np.asarray([], dtype=float)
    return arr


def array_int(data, key):
    if key not in data.files:
        return np.asarray([], dtype=np.int64)
    return np.asarray(data[key]).reshape(-1).astype(np.int64)


def mean_std(values):
    values = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    if not values:
        return None, None
    mean = sum(values) / float(len(values))
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / float(len(values) - 1)
    return mean, math.sqrt(var)


def median(values):
    values = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.median(values)) if values else None


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_mapping(array_id):
    seed_index = array_id % 5
    tmp = array_id // 5
    lambda_index = tmp % 4
    method_index = tmp // 4
    return METHODS[method_index], LAMBDAS[lambda_index], SEEDS[seed_index]


def result_path(method, lam, seed):
    return RUN_DIR / "{}_lambda_{}_seed_{}.npz".format(method, lam_tag(lam), seed)


def status_path(array_id, method, lam, seed):
    name = "case_{:03d}_{}_lambda_{}_seed_{}.status".format(
        array_id, method, lam_tag(lam), seed
    )
    return STATUS_DIR / name


def final_weights_for_candidates(data, method, n):
    if "candidate_final_alpha" in data.files:
        alpha = array_float(data, "candidate_final_alpha")
    elif "candidate_terminal_alpha" in data.files:
        alpha = array_float(data, "candidate_terminal_alpha")
    elif method == "fixed_equal":
        alpha = np.ones(n)
    elif method == "fixed_selected":
        alpha = np.ones(n)
    elif method == "lra":
        alpha = np.ones(n)
    else:
        alpha = np.full(n, np.nan)

    if "candidate_final_beta" in data.files:
        beta = array_float(data, "candidate_final_beta")
    elif "candidate_terminal_beta" in data.files:
        beta = array_float(data, "candidate_terminal_beta")
    elif method == "fixed_equal":
        beta = np.ones(n)
    elif method == "fixed_selected":
        beta = np.full(n, 3.0)
    else:
        beta = np.full(n, np.nan)
    return alpha, beta


selected_rows = []
candidate_rows = []
failures = []

for array_id in range(100):
    method, lam, seed = case_mapping(array_id)
    spath = status_path(array_id, method, lam, seed)
    rpath = result_path(method, lam, seed)
    status = parse_status(spath)

    state = "success"
    reason = ""
    if not spath.exists():
        state = "missing"
        reason = "status file missing"
    elif status.get("protocol") != EXPECTED_PROTOCOL:
        state = "stale_protocol"
        reason = "status protocol={}".format(status.get("protocol"))
    elif status.get("selection_protocol") != EXPECTED_SELECTION_PROTOCOL:
        state = "stale_protocol"
        reason = "status selection_protocol={}".format(status.get("selection_protocol"))
    elif status.get("exit_code") != "0":
        state = "failed"
        reason = "exit_code={}".format(status.get("exit_code"))
    elif not rpath.exists():
        state = "incomplete"
        reason = "result npz missing"

    base = {
        "array_id": array_id, "method": method, "lambda": lam, "seed": seed,
        "state": state, "reason": reason,
        "wall_time_seconds": status.get("wall_time_seconds"),
        "host": status.get("host"), "result_file": str(rpath),
        "status_file": str(spath),
    }

    if state != "success":
        selected_rows.append(base)
        failures.append(base)
        continue

    try:
        data = np.load(str(rpath), allow_pickle=False)
        protocol = scalar_text(data, "protocol_version")
        selection_protocol = scalar_text(data, "selection_protocol")
        if protocol != EXPECTED_PROTOCOL or selection_protocol != EXPECTED_SELECTION_PROTOCOL:
            base["state"] = "stale_protocol"
            base["reason"] = "npz protocol mismatch: {} / {}".format(protocol, selection_protocol)
            selected_rows.append(base); failures.append(base); data.close(); continue

        seeds_arr = array_int(data, "candidate_seed")
        vals = array_float(data, "candidate_validation_score")
        lf = array_float(data, "candidate_physics_loss")
        lb = array_float(data, "candidate_boundary_loss")
        times = array_float(data, "candidate_train_seconds")
        rel = array_float(data, "candidate_test_relative_l2")
        mse = array_float(data, "candidate_test_mse")
        mae = array_float(data, "candidate_test_mae")
        mx = array_float(data, "candidate_test_max_error")
        selected_flags = array_int(data, "candidate_selected")
        n = len(seeds_arr)

        required_lengths = [len(vals), len(lf), len(lb), len(times), len(rel), len(mse), len(mae), len(mx), len(selected_flags)]
        if n != EXPECTED_CANDIDATES or any(x != n for x in required_lengths):
            base["state"] = "incomplete"
            base["reason"] = "candidate arrays incomplete: n={} lengths={}".format(n, required_lengths)
            selected_rows.append(base); failures.append(base); data.close(); continue

        selected_candidate = int(np.asarray(data["selected_candidate"]).reshape(-1)[0])
        selected_idx = selected_candidate - 1
        selected_seed = int(np.asarray(data["selected_candidate_seed"]).reshape(-1)[0])
        if selected_idx < 0 or selected_idx >= n or selected_flags[selected_idx] != 1:
            base["state"] = "incomplete"
            base["reason"] = "invalid selected candidate metadata"
            selected_rows.append(base); failures.append(base); data.close(); continue

        cand_alpha, cand_beta = final_weights_for_candidates(data, method, n)
        if len(cand_alpha) != n or len(cand_beta) != n:
            cand_alpha = np.full(n, np.nan); cand_beta = np.full(n, np.nan)

        for j in range(n):
            candidate_rows.append({
                "array_id": array_id, "method": method, "lambda": lam,
                "outer_seed": seed, "candidate": j + 1,
                "candidate_seed": int(seeds_arr[j]),
                "selected": int(selected_flags[j]),
                "validation_physics": float(lf[j]),
                "validation_boundary": float(lb[j]),
                "validation_score": float(vals[j]),
                "test_relative_l2": float(rel[j]),
                "test_mse": float(mse[j]), "test_mae": float(mae[j]),
                "test_max_error": float(mx[j]),
                "final_alpha": float(cand_alpha[j]),
                "final_beta": float(cand_beta[j]),
                "train_seconds": float(times[j]),
            })

        base.update({
            "protocol_version": protocol,
            "selection_protocol": selection_protocol,
            "n_candidates": n,
            "selected_candidate": selected_candidate,
            "selected_candidate_seed": selected_seed,
            "validation_score": float(vals[selected_idx]),
            "relative_l2": float(rel[selected_idx]),
            "mse": float(mse[selected_idx]), "mae": float(mae[selected_idx]),
            "max_error": float(mx[selected_idx]),
            "alpha": float(cand_alpha[selected_idx]),
            "beta": float(cand_beta[selected_idx]),
            "candidate_train_seconds_total": float(np.sum(times)),
            "candidate_train_seconds_mean": float(np.mean(times)),
        })
        selected_rows.append(base)
        data.close()
    except Exception as exc:
        base["state"] = "incomplete"
        base["reason"] = "npz read error: {}".format(exc)
        selected_rows.append(base); failures.append(base)

# Stable field order; DictWriter fills missing values with blanks.
selected_fields = [
    "array_id", "method", "lambda", "seed", "state", "reason",
    "protocol_version", "selection_protocol", "n_candidates",
    "selected_candidate", "selected_candidate_seed", "validation_score",
    "relative_l2", "mse", "mae", "max_error", "alpha", "beta",
    "candidate_train_seconds_total", "candidate_train_seconds_mean",
    "wall_time_seconds", "host", "result_file", "status_file",
]
for row in selected_rows:
    for key in selected_fields:
        row.setdefault(key, None)
write_csv(FINAL_DIR / "final_results.csv", selected_rows, selected_fields)

candidate_fields = [
    "array_id", "method", "lambda", "outer_seed", "candidate",
    "candidate_seed", "selected", "validation_physics", "validation_boundary",
    "validation_score", "test_relative_l2", "test_mse", "test_mae",
    "test_max_error", "final_alpha", "final_beta", "train_seconds",
]
write_csv(FINAL_DIR / "candidate_results.csv", candidate_rows, candidate_fields)

# ---------------------------------------------------------------------------
# Selected-result summaries
# ---------------------------------------------------------------------------
summary_rows = []
for method in METHODS:
    for lam in LAMBDAS:
        group = [r for r in selected_rows if r["state"] == "success" and r["method"] == method and float(r["lambda"]) == lam]
        rel_m, rel_s = mean_std([r.get("relative_l2") for r in group])
        mse_m, mse_s = mean_std([r.get("mse") for r in group])
        val_m, val_s = mean_std([r.get("validation_score") for r in group])
        tim_m, tim_s = mean_std([r.get("candidate_train_seconds_total") for r in group])
        summary_rows.append({
            "method": method, "lambda": lam, "n_success": len(group),
            "relative_l2_mean": rel_m, "relative_l2_std": rel_s,
            "relative_l2_median": median([r.get("relative_l2") for r in group]),
            "mse_mean": mse_m, "mse_std": mse_s,
            "validation_score_mean": val_m, "validation_score_std": val_s,
            "candidate_train_seconds_total_mean": tim_m,
            "candidate_train_seconds_total_std": tim_s,
        })
summary_fields = [
    "method", "lambda", "n_success", "relative_l2_mean", "relative_l2_std",
    "relative_l2_median", "mse_mean", "mse_std", "validation_score_mean",
    "validation_score_std", "candidate_train_seconds_total_mean",
    "candidate_train_seconds_total_std",
]
write_csv(FINAL_DIR / "method_summary.csv", summary_rows, summary_fields)

# All-candidate post-selection audit summary. This does NOT affect selection.
candidate_summary_rows = []
for method in METHODS:
    for lam in LAMBDAS:
        group = [r for r in candidate_rows if r["method"] == method and float(r["lambda"]) == lam]
        rel_m, rel_s = mean_std([r["test_relative_l2"] for r in group])
        val_m, val_s = mean_std([r["validation_score"] for r in group])
        tim_m, tim_s = mean_std([r["train_seconds"] for r in group])
        candidate_summary_rows.append({
            "method": method, "lambda": lam, "n_candidates": len(group),
            "test_relative_l2_mean": rel_m, "test_relative_l2_std": rel_s,
            "test_relative_l2_median": median([r["test_relative_l2"] for r in group]),
            "validation_score_mean": val_m, "validation_score_std": val_s,
            "train_seconds_mean": tim_m, "train_seconds_std": tim_s,
        })
candidate_summary_fields = [
    "method", "lambda", "n_candidates", "test_relative_l2_mean",
    "test_relative_l2_std", "test_relative_l2_median", "validation_score_mean",
    "validation_score_std", "train_seconds_mean", "train_seconds_std",
]
write_csv(FINAL_DIR / "candidate_method_summary.csv", candidate_summary_rows, candidate_summary_fields)

# ---------------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------------
def average_tie_ranks(items):
    items = sorted(items, key=lambda x: x[1])
    out = []
    i = 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j][1] == items[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            out.append((items[k][0], items[k][1], rank))
        i = j
    return out


def make_rank_rows(methods):
    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            group = [r for r in selected_rows if r["state"] == "success" and r["method"] in methods and float(r["lambda"]) == lam and int(r["seed"]) == seed]
            if len(group) != len(methods):
                continue
            for method, value, rank in average_tie_ranks([(r["method"], float(r["relative_l2"])) for r in group]):
                rows.append({"lambda": lam, "seed": seed, "method": method, "relative_l2": value, "rank": rank})
    return rows


def rank_summary(rank_rows, methods):
    out = []
    for method in methods:
        ranks = [float(r["rank"]) for r in rank_rows if r["method"] == method]
        out.append({
            "method": method, "n_ranked_cases": len(ranks),
            "mean_rank": (sum(ranks) / float(len(ranks))) if ranks else None,
            "wins": sum(1 for x in ranks if x == 1.0),
        })
    return out

rank_fields = ["lambda", "seed", "method", "relative_l2", "rank"]
all_ranks = make_rank_rows(METHODS)
adaptive_ranks = make_rank_rows(ADAPTIVE_METHODS)
write_csv(FINAL_DIR / "case_ranks.csv", all_ranks, rank_fields)
write_csv(FINAL_DIR / "adaptive_case_ranks.csv", adaptive_ranks, rank_fields)
rank_summary_fields = ["method", "n_ranked_cases", "mean_rank", "wins"]
write_csv(FINAL_DIR / "method_rank_summary.csv", rank_summary(all_ranks, METHODS), rank_summary_fields)
write_csv(FINAL_DIR / "adaptive_method_rank_summary.csv", rank_summary(adaptive_ranks, ADAPTIVE_METHODS), rank_summary_fields)

# ---------------------------------------------------------------------------
# Matched non-parametric statistics: two-sided, no method is privileged.
# ---------------------------------------------------------------------------
def matched_vectors(methods):
    vectors = {m: [] for m in methods}
    cases = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            lookup = {(r["method"]): float(r["relative_l2"]) for r in selected_rows if r["state"] == "success" and r["method"] in methods and float(r["lambda"]) == lam and int(r["seed"]) == seed}
            if all(m in lookup for m in methods):
                cases.append((lam, seed))
                for m in methods:
                    vectors[m].append(lookup[m])
    return cases, vectors


def holm_adjust(p_values):
    # Returns adjusted p-values in original order.
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted

stat_rows = []
for scope, methods in [("all_methods", METHODS), ("adaptive_methods", ADAPTIVE_METHODS)]:
    cases, vectors = matched_vectors(methods)
    n_cases = len(cases)
    if friedmanchisquare is not None and n_cases > 0:
        stat, p = friedmanchisquare(*[vectors[m] for m in methods])
        stat_rows.append({
            "scope": scope, "test": "Friedman", "comparison": "all",
            "n_cases": n_cases, "statistic": float(stat), "p_value": float(p),
            "p_holm": None,
        })
    else:
        stat_rows.append({
            "scope": scope, "test": "Friedman", "comparison": "all",
            "n_cases": n_cases, "statistic": None, "p_value": None, "p_holm": None,
        })

    pairs = list(itertools.combinations(methods, 2))
    pair_stats = []
    pvals = []
    if wilcoxon is not None and n_cases > 0:
        for a, b in pairs:
            try:
                stat, p = wilcoxon(vectors[a], vectors[b], alternative="two-sided")
                pair_stats.append((a, b, float(stat), float(p)))
                pvals.append(float(p))
            except Exception:
                pair_stats.append((a, b, None, 1.0)); pvals.append(1.0)
        adjusted = holm_adjust(pvals)
        for item, p_adj in zip(pair_stats, adjusted):
            a, b, stat, p = item
            stat_rows.append({
                "scope": scope, "test": "Wilcoxon_two_sided",
                "comparison": "{} vs {}".format(a, b), "n_cases": n_cases,
                "statistic": stat, "p_value": p, "p_holm": p_adj,
            })

stat_fields = ["scope", "test", "comparison", "n_cases", "statistic", "p_value", "p_holm"]
write_csv(FINAL_DIR / "statistical_tests.csv", stat_rows, stat_fields)

failed_path = FINAL_DIR / "failed_or_missing_cases.txt"
with failed_path.open("w", encoding="utf-8") as handle:
    if not failures:
        handle.write("All 100 benchmark-v2 cases completed successfully.\n")
    else:
        for r in failures:
            handle.write("{array_id:03d} | {state:14s} | {method:14s} | lambda={lambda} | seed={seed} | {reason}\n".format(**r))

counts = {}
for row in selected_rows:
    counts[row["state"]] = counts.get(row["state"], 0) + 1
print("=" * 78)
print("BRATU ADAPTIVE BENCHMARK V2 - COLLECTION SUMMARY")
print("=" * 78)
for state in ["success", "failed", "stale_protocol", "incomplete", "missing"]:
    print("{:<16}: {}".format(state, counts.get(state, 0)))
print("Candidate rows  : {} / 500".format(len(candidate_rows)))
print("Output directory: {}".format(FINAL_DIR))
print("Files: final_results.csv, method_summary.csv, candidate_results.csv,")
print("       candidate_method_summary.csv, case_ranks.csv,")
print("       adaptive_case_ranks.csv, method_rank_summary.csv,")
print("       adaptive_method_rank_summary.csv, statistical_tests.csv,")
print("       failed_or_missing_cases.txt")
print("=" * 78)

if counts.get("success", 0) != 100:
    raise SystemExit(1)
