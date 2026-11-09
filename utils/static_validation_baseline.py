"""Validation-only static-weight selection protocol for the Bratu benchmark.

This module is additive: original benchmark files/outputs are not modified.
Python 3.7 compatible, including the historical Cluster runtime.
"""
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from configs import bratu_config as cfg

PROTOCOL = "bratu_validation_only_static_v1"
METHOD = "fixed_validation"
WEIGHTS = ((1.0, 0.1), (1.0, 0.3), (1.0, 1.0), (1.0, 3.0), (1.0, 10.0))
LAMBDAS = (1.0, 2.0, 3.0, 3.4)
DEV_SEEDS = (2027, 2029, 2039)
FINAL_SEEDS = (42, 123, 456, 789, 1000)
SMOKE_KEYS = ("BRATU_TOTAL_PINN_STEPS", "BRATU_N_COLLOCATION", "BRATU_N_VALIDATION", "BRATU_N_TEST", "BRATU_SELECTION_CANDIDATES")
SOURCE_FILES = (
    "configs/bratu_config.py", "models/pinn.py", "problems/bratu.py",
    "training/trainer.py", "utils/bratu_eval.py", "utils/fair_selection.py",
    "utils/static_validation_baseline.py",
    "experiments/bratu_static_validation_tune.py",
    "experiments/bratu_fixed_validation.py",
)


def tune_cases():
    return [(index, alpha, beta, lam, seed)
            for index, (alpha, beta, lam, seed) in enumerate(
                (a, b, lam, seed)
                for a, b in WEIGHTS
                for lam in LAMBDAS
                for seed in DEV_SEEDS)]


def final_cases():
    return [(index, lam, seed)
            for index, (lam, seed) in enumerate(
                (lam, seed) for lam in LAMBDAS for seed in FINAL_SEEDS)]


def config_snapshot():
    return {
        "architecture": list(cfg.HIDDEN_LAYERS),
        "learning_rate": float(cfg.LEARNING_RATE),
        "n_collocation": int(cfg.N_COLLOCATION),
        "n_validation": int(cfg.N_VALIDATION),
        "n_test": int(cfg.N_TEST),
        "total_pinn_steps": int(cfg.TOTAL_PINN_STEPS),
        "selection_candidates": int(cfg.SELECTION_CANDIDATES),
        "candidate_seed_stride": int(cfg.CANDIDATE_SEED_STRIDE),
        "candidate_selector": "min_unweighted_Lf_val_plus_Lb_val",
        "underlying_benchmark": "bratu_adaptive_benchmark_v2",
    }


def validate_config(smoke):
    c = config_snapshot()
    if not smoke:
        expected = {
            "architecture": [64, 64, 64], "learning_rate": 0.001,
            "n_collocation": 256, "n_validation": 512,
            "n_test": 2000, "total_pinn_steps": 15000,
            "selection_candidates": 5, "candidate_seed_stride": 1000003,
        }
        for key, value in expected.items():
            if c[key] != value:
                raise ValueError("Production protocol mismatch: {}={} != {}. "
                                 "Use --smoke for reduced settings.".format(key, c[key], value))
    for key in ("n_collocation", "n_validation", "n_test", "total_pinn_steps", "selection_candidates"):
        if c[key] < 1:
            raise ValueError("Invalid configuration: {}".format(key))
    return c


def source_fingerprint(project_root):
    checksums = {}
    for relative in SOURCE_FILES:
        file_path = Path(project_root) / relative
        if not file_path.is_file():
            raise IOError("Missing required source file: {}".format(file_path))
        checksums[relative] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return checksums


def environment_snapshot(device):
    return {
        "python": sys.version.split("\n")[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": int(torch.get_num_threads()),
        "pbs_job_id": os.environ.get("PBS_JOBID", ""),
        "pbs_array_id": os.environ.get("PBS_ARRAYID", os.environ.get("PBS_ARRAY_INDEX", "")),
    }


def result_root(project_root, smoke=False, base_dir=None):
    root = Path(base_dir) if base_dir else Path(project_root) / "results" / "validation_static_v1"
    return root / ("smoke" if smoke else "production")


def device_choice(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in ("cuda", "cpu"):
        raise ValueError("Device must be auto/cpu/cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return value


def atomic_json_write(path, data):
    """Never overwrite existing results; same-directory hard-link is atomic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".staging_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, str(path))  # O_EXCL semantics; fail rather than overwrite
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_npz_write(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".staging_", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, str(path))
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_same(expected, actual, label):
    if expected != actual:
        raise ValueError("{} mismatch. Refusing to combine incompatible runs.".format(label))


def summarize_validation_trials(trials):
    """Select only by average log of unweighted validation scores.

    Never reads a reference prediction, reference metric, or test field.
    """
    scores = []
    for alpha, beta in WEIGHTS:
        matches = [r for r in trials if r["alpha"] == alpha and r["beta"] == beta]
        if len(matches) != len(LAMBDAS) * len(DEV_SEEDS):
            raise ValueError("Incomplete tuning grid for ({},{})".format(alpha, beta))
        logs = [math.log(max(float(r["validation_score"]), 1e-30)) for r in matches]
        scores.append({"alpha": alpha, "beta": beta,
                       "mean_log_validation_score": float(sum(logs) / len(logs)),
                       "geometric_mean_validation_score": float(math.exp(sum(logs) / len(logs))),
                       "mean_validation_score": float(np.mean([r["validation_score"] for r in matches])),
                       "std_validation_score": float(np.std([r["validation_score"] for r in matches], ddof=1))})
    # Python min is stable: on equal objective, first grid pair wins.
    winner = min(scores, key=lambda row: row["mean_log_validation_score"])
    return winner, scores
