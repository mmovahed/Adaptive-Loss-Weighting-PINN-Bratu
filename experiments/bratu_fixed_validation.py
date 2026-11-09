"""Held-out best-of-five PINN evaluation of a previously locked static pair.

Run ONLY after bratu_static_validation_tune.py --finalize.
--index 0..19 enumerates four lambdas x five historical outer seeds.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import bratu_config as cfg
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from utils.bratu_eval import (
    set_seed, make_validation_grid, validation_losses,
    evaluate_candidate_models, print_result_json,
)
from utils.fair_selection import make_candidate_seeds, select_best_candidate
from utils.static_validation_baseline import (
    METHOD, PROTOCOL, final_cases, config_snapshot, validate_config,
    source_fingerprint, environment_snapshot, result_root, device_choice,
    atomic_npz_write, atomic_json_write, read_json, assert_same,
)


def load_locked_pair(root, smoke, config, source):
    lock_path = root / "selected_weights.json"
    if not lock_path.is_file():
        raise RuntimeError("No locked validation-only weights. First finish 60 tuning tasks "
                           "and run the tuner with --finalize.")
    content = lock_path.read_bytes()
    lock = json.loads(content.decode("utf-8"))
    assert_same(lock["protocol"], PROTOCOL, "locked protocol")
    assert_same(lock["phase"], "locked_validation_only", "lock phase")
    assert_same(lock["smoke"], smoke, "smoke/production namespace")
    assert_same(lock["config"], config, "training/validation config")
    assert_same(lock["source_sha256"], source, "source fingerprint")
    if lock["n_tuning_fits"] != 60:
        raise ValueError("Lock does not record all 60 tuning fits")
    return lock, hashlib.sha256(content).hexdigest()


def train_candidate(problem, validation_grid, device, seed, alpha, beta):
    set_seed(seed)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=cfg.HIDDEN_LAYERS).to(device)
    trainer = PINNTrainer(model, problem, learning_rate=cfg.LEARNING_RATE,
                          physics_weight=alpha, boundary_weight=beta, device=device)
    start = time.perf_counter()
    history = trainer.train_adam(
        epochs=cfg.TOTAL_PINN_STEPS, n_collocation=cfg.N_COLLOCATION,
        print_every=max(1000, cfg.TOTAL_PINN_STEPS + 1),
    )
    seconds = time.perf_counter() - start
    lf_val, lb_val = validation_losses(model, problem, validation_grid)
    score = lf_val + lb_val
    if not (math.isfinite(score) and score > 0):
        raise ValueError("Invalid validation score: {}".format(score))
    return model, history, lf_val, lb_val, seconds


def run_index(index, root, device, smoke, config, source):
    cases = final_cases()
    if not 0 <= index < len(cases):
        raise ValueError("Evaluation index out of range: {}".format(index))
    lock, lock_sha256 = load_locked_pair(root, smoke, config, source)
    alpha, beta = lock["selected_alpha"], lock["selected_beta"]
    _, lam, outer_seed = cases[index]
    tag = "fixed_validation_lambda_{}_seed_{}".format(str(lam).replace(".", "p"), outer_seed)
    path = root / "evaluation" / (tag + ".npz")
    summary_path = root / "evaluation" / (tag + ".json")
    if path.exists() or summary_path.exists():
        if not path.exists() or not summary_path.exists():
            raise RuntimeError("Partial prior output for {}. Inspect manually.".format(tag))
        previous = read_json(summary_path)
        assert_same(previous["lock_sha256"], lock_sha256, "evaluation lock")
        assert_same(previous["config"], config, "evaluation config")
        print("Already complete (not overwritten): {}".format(path))
        return
    print("EVALUATE {} / {} | lambda={} outer_seed={} locked weights=({}, {})".format(
        index + 1, len(cases), lam, outer_seed, alpha, beta), flush=True)
    set_seed(outer_seed)
    seeds = make_candidate_seeds(outer_seed, cfg.SELECTION_CANDIDATES, cfg.CANDIDATE_SEED_STRIDE)
    problem = BratuProblem(lam=lam)
    grid = make_validation_grid(cfg.N_VALIDATION, device)
    models, histories, lfs, lbs, times, scores = [], [], [], [], [], []
    for j, seed in enumerate(seeds, 1):
        model, history, lf, lb, sec = train_candidate(problem, grid, device, seed, alpha, beta)
        models.append(model); histories.append(history); lfs.append(lf); lbs.append(lb)
        times.append(sec); scores.append(lf + lb)
        print("  candidate {} seed={} | Lf={:.7e} Lb={:.7e} S_val={:.7e} time={:.2f}s".format(
            j, seed, lf, lb, lf + lb, sec), flush=True)
    # This selection MUST happen before any reference/test call.
    best = select_best_candidate(scores)
    selected_seed = seeds[best]
    selected_candidate = best + 1
    print("Validation-only selection FROZEN: candidate {} seed={}".format(
        selected_candidate, selected_seed), flush=True)
    # Only now may reference solutions and test errors be computed.
    candidate_results = evaluate_candidate_models(models, problem, cfg.N_TEST, device)
    result = candidate_results[best]
    arrays = {
        "protocol_version": np.asarray([PROTOCOL]),
        "selection_protocol": np.asarray([cfg.SELECTION_PROTOCOL]),
        "method": np.asarray([METHOD]), "lambda": np.asarray([lam]),
        "outer_seed": np.asarray([outer_seed], dtype=np.int64),
        "tuning_lock_sha256": np.asarray([lock_sha256]),
        "alpha": np.asarray([alpha]), "beta": np.asarray([beta]),
        "x": result["x"], "y_pred": result["y_pred"], "y_ref": result["y_ref"],
        "history_total": np.asarray(histories[best]["total_loss"]),
        "history_physics": np.asarray(histories[best]["physics_loss"]),
        "history_boundary": np.asarray(histories[best]["boundary_loss"]),
        "validation_score": np.asarray([scores[best]]),
        "candidate_seed": np.asarray(seeds, dtype=np.int64),
        "candidate_physics_loss": np.asarray(lfs),
        "candidate_boundary_loss": np.asarray(lbs),
        "candidate_validation_score": np.asarray(scores),
        "candidate_train_seconds": np.asarray(times),
        "candidate_test_relative_l2": np.asarray([r["relative_l2"] for r in candidate_results]),
        "candidate_test_mse": np.asarray([r["mse"] for r in candidate_results]),
        "candidate_test_mae": np.asarray([r["mae"] for r in candidate_results]),
        "candidate_test_max_error": np.asarray([r["max_error"] for r in candidate_results]),
        "candidate_selected": (np.arange(len(seeds)) == best).astype(np.int8),
        "selected_candidate": np.asarray([selected_candidate], dtype=np.int64),
        "selected_candidate_seed": np.asarray([selected_seed], dtype=np.int64),
    }
    # Commit the .npz result, then JSON index (will detect partial output on rerun).
    atomic_npz_write(path, arrays)
    summary = {
        "protocol": PROTOCOL, "phase": "evaluation", "method": METHOD,
        "smoke": smoke, "index": index, "lambda": lam, "outer_seed": outer_seed,
        "alpha": alpha, "beta": beta, "selected_candidate": selected_candidate,
        "selected_candidate_seed": selected_seed, "n_candidates": len(seeds),
        "validation_score": float(scores[best]),
        "physics_validation_loss": float(lfs[best]),
        "boundary_validation_loss": float(lbs[best]),
        "mse": result["mse"], "mae": result["mae"],
        "max_error": result["max_error"], "relative_l2": result["relative_l2"],
        "candidate_train_seconds_total": float(sum(times)),
        "candidate_train_seconds_mean": float(np.mean(times)),
        "tuning_train_seconds_total_global": lock["tuning_train_seconds_total"],
        "lock_sha256": lock_sha256, "config": config,
        "source_sha256": source, "environment": environment_snapshot(device),
        "npz": path.name,
    }
    atomic_json_write(summary_path, summary)
    print_result_json(summary)
    print("Saved: {}".format(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Evaluation task index 0..19")
    group.add_argument("--all", action="store_true", help="Run each missing evaluation sequentially")
    parser.add_argument("--smoke", action="store_true", help="NON-publication smoke-test namespace")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    config = validate_config(args.smoke)
    root = result_root(PROJECT_ROOT, args.smoke, args.output_root)
    source = source_fingerprint(PROJECT_ROOT)
    device = device_choice(args.device)
    if args.all:
        for idx in range(len(final_cases())):
            run_index(idx, root, device, args.smoke, config, source)
    else:
        run_index(args.index, root, device, args.smoke, config, source)


if __name__ == "__main__":
    main()
