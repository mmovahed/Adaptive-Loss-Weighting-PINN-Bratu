"""Development-only validation search; NEVER consults a Bratu reference solution.

Run --index 0..59 independently (e.g., as a PBS array), then run --finalize.
For local smoke tests use --smoke and environment variable step/point overrides.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import bratu_config as cfg
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from utils.bratu_eval import make_validation_grid, set_seed, validation_losses
from utils.static_validation_baseline import (
    PROTOCOL, WEIGHTS, LAMBDAS, DEV_SEEDS, tune_cases, config_snapshot,
    validate_config, source_fingerprint, environment_snapshot, result_root,
    device_choice, atomic_json_write, read_json, assert_same,
    summarize_validation_trials,
)


def train_one(alpha, beta, lam, seed, device):
    """One full-budget PINN, assessed ONLY on validation residual/BC."""
    set_seed(seed)
    problem = BratuProblem(lam=lam)
    grid = make_validation_grid(cfg.N_VALIDATION, device)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=cfg.HIDDEN_LAYERS).to(device)
    trainer = PINNTrainer(
        model, problem, learning_rate=cfg.LEARNING_RATE,
        physics_weight=alpha, boundary_weight=beta, device=device,
    )
    t0 = time.perf_counter()
    trainer.train_adam(
        epochs=cfg.TOTAL_PINN_STEPS,
        n_collocation=cfg.N_COLLOCATION,
        print_every=max(1000, cfg.TOTAL_PINN_STEPS + 1),
    )
    seconds = time.perf_counter() - t0
    lf_val, lb_val = validation_losses(model, problem, grid)
    score = lf_val + lb_val
    if not (math.isfinite(score) and score > 0.0):
        raise ValueError("Non-positive/non-finite validation score")
    return {"physics_validation_loss": lf_val, "boundary_validation_loss": lb_val,
            "validation_score": score, "train_seconds": seconds}


def run_index(index, root, device, smoke, config, source):
    cases = tune_cases()
    if not 0 <= index < len(cases):
        raise ValueError("Tuning index out of range: {}".format(index))
    _, alpha, beta, lam, seed = cases[index]
    path = root / "tuning" / "trial_{:03d}.json".format(index)
    if path.exists():
        old = read_json(path)
        assert_same(old["config"], config, "saved tuning config")
        assert_same(old["source_sha256"], source, "saved tuning source")
        assert_same([old["alpha"], old["beta"], old["lambda"], old["seed"]],
                    [alpha, beta, lam, seed], "saved tuning case")
        print("Already complete (not overwritten): {}".format(path))
        return
    print("TUNE {} / {} | alpha={}, beta={}, lambda={}, seed={}".format(
        index + 1, len(cases), alpha, beta, lam, seed), flush=True)
    metrics = train_one(alpha, beta, lam, seed, device)
    record = dict(metrics, protocol=PROTOCOL, phase="tuning", smoke=smoke,
                  index=index, alpha=alpha, beta=beta, **{"lambda": lam}, seed=seed,
                  config=config, source_sha256=source, environment=environment_snapshot(device))
    atomic_json_write(path, record)
    print("Validated: S_val={:.8e}; saved {}".format(metrics["validation_score"], path), flush=True)


def finalize(root, smoke, config, source):
    cases = tune_cases()
    trials = []
    for idx, alpha, beta, lam, seed in cases:
        path = root / "tuning" / "trial_{:03d}.json".format(idx)
        if not path.exists():
            raise RuntimeError("Incomplete tuning: missing {}. No weights locked.".format(path))
        row = read_json(path)
        assert_same(row["protocol"], PROTOCOL, "protocol")
        assert_same(row["phase"], "tuning", "phase")
        assert_same(row["smoke"], smoke, "smoke flag")
        assert_same(row["config"], config, "config")
        assert_same(row["source_sha256"], source, "source fingerprint")
        assert_same([row["index"], row["alpha"], row["beta"], row["lambda"], row["seed"]],
                    [idx, alpha, beta, lam, seed], "trial identity")
        if not (math.isfinite(row["validation_score"]) and row["validation_score"] > 0):
            raise ValueError("Invalid validation score in trial {}".format(idx))
        trials.append(row)
    winner, grid_summary = summarize_validation_trials(trials)
    lock = {
        "protocol": PROTOCOL, "phase": "locked_validation_only", "smoke": smoke,
        "tuning_rule": "argmin mean(log(max(Lf_val+Lb_val,1e-30))) across 4 lambdas x 3 dev seeds",
        "tie_breaker": "first listed pair", "search_grid": [list(x) for x in WEIGHTS],
        "development_lambdas": list(LAMBDAS), "development_seeds": list(DEV_SEEDS),
        "final_evaluation_seeds": [42, 123, 456, 789, 1000],
        "selected_alpha": winner["alpha"], "selected_beta": winner["beta"],
        "selected_objective": winner["mean_log_validation_score"],
        "grid_summary": grid_summary, "n_tuning_fits": len(trials),
        "tuning_train_seconds_total": float(sum(x["train_seconds"] for x in trials)),
        "config": config, "source_sha256": source,
        "trial_paths": ["tuning/trial_{:03d}.json".format(i) for i in range(len(trials))],
    }
    path = root / "selected_weights.json"
    atomic_json_write(path, lock)
    print("LOCKED validation-only pair: ({}, {})".format(winner["alpha"], winner["beta"]))
    print("Tuning training seconds (sum): {:.3f}".format(lock["tuning_train_seconds_total"]))
    print("Lock: {}".format(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Tuning task index, 0..59")
    group.add_argument("--all", action="store_true", help="Run every missing task sequentially")
    group.add_argument("--finalize", action="store_true", help="Lock pair after all 60 tasks")
    parser.add_argument("--smoke", action="store_true", help="Separate NON-publication test namespace")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default=None, help="Optional base results directory")
    args = parser.parse_args()
    config = validate_config(args.smoke)
    root = result_root(PROJECT_ROOT, args.smoke, args.output_root)
    source = source_fingerprint(PROJECT_ROOT)
    if args.finalize:
        finalize(root, args.smoke, config, source)
    else:
        device = device_choice(args.device)
        if args.all:
            for idx in range(len(tune_cases())):
                run_index(idx, root, device, args.smoke, config, source)
        else:
            run_index(args.index, root, device, args.smoke, config, source)


if __name__ == "__main__":
    main()
