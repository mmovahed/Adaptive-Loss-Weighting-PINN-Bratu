"""Collect validation-only static-weight study without touching legacy results.

Requires all 60 tuning files, a locked pair, and all 20 final cases.
Creates four CSV tables in a new summary/ directory (no overwrite).
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.static_validation_baseline import (
    LAMBDAS, DEV_SEEDS, FINAL_SEEDS, PROTOCOL, WEIGHTS,
    config_snapshot, validate_config, source_fingerprint, result_root,
    tune_cases, final_cases, read_json, assert_same,
)


def write_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(root, smoke, config, source):
    lock_path = root / "selected_weights.json"
    if not lock_path.exists():
        raise RuntimeError("Tuning is not finalized")
    raw_lock = lock_path.read_bytes()
    lock_sha = hashlib.sha256(raw_lock).hexdigest()
    lock = json.loads(raw_lock.decode("utf-8"))
    assert_same(lock["protocol"], PROTOCOL, "protocol")
    assert_same(lock["smoke"], smoke, "smoke mode")
    assert_same(lock["config"], config, "config")
    assert_same(lock["source_sha256"], source, "source")
    tuning = []
    for index, alpha, beta, lam, seed in tune_cases():
        path = root / "tuning" / "trial_{:03d}.json".format(index)
        record = read_json(path)
        assert_same([record["alpha"], record["beta"], record["lambda"], record["seed"]],
                    [alpha, beta, lam, seed], "tuning identity")
        assert_same(record["source_sha256"], source, "tuning source")
        tuning.append(record)
    final_rows = []
    candidate_rows = []
    for index, lam, seed in final_cases():
        tag = "fixed_validation_lambda_{}_seed_{}".format(str(lam).replace(".", "p"), seed)
        summary = read_json(root / "evaluation" / (tag + ".json"))
        path = root / "evaluation" / (tag + ".npz")
        assert_same(summary["lock_sha256"], lock_sha, "evaluation lock")
        assert_same(summary["source_sha256"], source, "evaluation source")
        assert_same([summary["index"], summary["lambda"], summary["outer_seed"]],
                    [index, lam, seed], "evaluation identity")
        assert_same([summary["alpha"], summary["beta"]],
                    [lock["selected_alpha"], lock["selected_beta"]], "selected weights")
        with np.load(str(path), allow_pickle=False) as data:
            val = data["candidate_validation_score"]
            chosen = int(data["selected_candidate"][0]) - 1
            if chosen != int(np.argmin(val)):
                raise ValueError("Candidate selection mismatch in {}".format(path))
            if int(data["candidate_seed"][chosen]) != summary["selected_candidate_seed"]:
                raise ValueError("Selected candidate seed mismatch in {}".format(path))
            for j in range(len(val)):
                candidate_rows.append({
                    "lambda": lam, "outer_seed": seed, "candidate": j + 1,
                    "candidate_seed": int(data["candidate_seed"][j]),
                    "selected": int(data["candidate_selected"][j]),
                    "validation_score": float(val[j]),
                    "physics_validation_loss": float(data["candidate_physics_loss"][j]),
                    "boundary_validation_loss": float(data["candidate_boundary_loss"][j]),
                    "relative_l2": float(data["candidate_test_relative_l2"][j]),
                    "mse": float(data["candidate_test_mse"][j]),
                    "mae": float(data["candidate_test_mae"][j]),
                    "max_error": float(data["candidate_test_max_error"][j]),
                    "train_seconds": float(data["candidate_train_seconds"][j]),
                })
        final_rows.append(summary)
    by_lambda = []
    for lam in LAMBDAS:
        items = [r for r in final_rows if r["lambda"] == lam]
        if len(items) != len(FINAL_SEEDS):
            raise ValueError("Incomplete lambda group {}".format(lam))
        errors = np.asarray([r["relative_l2"] for r in items], dtype=float)
        seconds = np.asarray([r["candidate_train_seconds_total"] for r in items], dtype=float)
        by_lambda.append({"lambda": lam, "n_seeds": len(items),
                          "relative_l2_mean": float(np.mean(errors)),
                          "relative_l2_sd": float(np.std(errors, ddof=1)),
                          "relative_l2_median": float(np.median(errors)),
                          "training_seconds_mean": float(np.mean(seconds)),
                          "training_seconds_sd": float(np.std(seconds, ddof=1))})
    grid_rows = lock["grid_summary"]
    summary_root = root / "summary"
    outputs = [
        (summary_root / "tuning_trials.csv", tuning,
         ["index", "alpha", "beta", "lambda", "seed", "physics_validation_loss", "boundary_validation_loss", "validation_score", "train_seconds"]),
        (summary_root / "tuning_grid.csv", grid_rows,
         ["alpha", "beta", "mean_log_validation_score", "geometric_mean_validation_score", "mean_validation_score", "std_validation_score"]),
        (summary_root / "selected_cases.csv", final_rows,
         ["method", "lambda", "outer_seed", "alpha", "beta", "selected_candidate", "selected_candidate_seed", "validation_score", "physics_validation_loss", "boundary_validation_loss", "relative_l2", "mse", "mae", "max_error", "candidate_train_seconds_total", "tuning_train_seconds_total_global", "lock_sha256"]),
        (summary_root / "all_candidates.csv", candidate_rows,
         ["lambda", "outer_seed", "candidate", "candidate_seed", "selected", "validation_score", "physics_validation_loss", "boundary_validation_loss", "relative_l2", "mse", "mae", "max_error", "train_seconds"]),
        (summary_root / "by_lambda.csv", by_lambda,
         ["lambda", "n_seeds", "relative_l2_mean", "relative_l2_sd", "relative_l2_median", "training_seconds_mean", "training_seconds_sd"]),
    ]
    for path, rows, fields in outputs:
        if path.exists():
            raise FileExistsError("Cannot overwrite summary: {}".format(path))
    for path, rows, fields in outputs:
        write_csv_new(path, rows, fields)
        print("Wrote {:3d} rows to {}".format(len(rows), path))
    print("Tuning overhead: {:.2f} training seconds across 60 development fits".format(
        lock["tuning_train_seconds_total"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    cfg = validate_config(args.smoke)
    root = result_root(PROJECT_ROOT, args.smoke, args.output_root)
    collect(root, args.smoke, cfg, source_fingerprint(PROJECT_ROOT))


if __name__ == "__main__":
    main()
