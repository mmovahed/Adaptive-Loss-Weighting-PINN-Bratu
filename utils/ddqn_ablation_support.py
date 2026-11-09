"""Narrow, additive helpers for the Bratu DDQN normalization ablation.
Compatible with Python 3.7 / NumPy 1.19.
"""
import json
import hashlib
import os
import tempfile
from pathlib import Path


def normalized_action(alpha, beta, action, multipliers, min_weight=0.01):
    """Apply the same 5x5 DDQN action, then project weights to alpha+beta=2.

    Projection preserves the proposed relative multiplicative action, except at
    the positivity floor. This is only called in the normalized treatment.
    """
    if int(action) != action or action < 0 or action >= 25:
        raise ValueError("DDQN action must be an integer in [0,24]")
    if alpha <= 0.0 or beta <= 0.0 or not (0 < min_weight < 1):
        raise ValueError("positive coefficients and 0<min_weight<1 required")
    a = float(alpha) * float(multipliers[int(action) // 5])
    b = float(beta) * float(multipliers[int(action) % 5])
    new_a = 2.0 * a / (a + b)
    new_a = max(float(min_weight), min(2.0 - float(min_weight), new_a))
    return float(new_a), float(2.0 - new_a)


def decode_array_index(index):
    index = int(index)
    if index < 0 or index >= 40:
        raise ValueError("Array index must be 0..39")
    modes = ("original_control", "normalized")
    lambdas = (1.0, 2.0, 3.0, 3.4)
    seeds = (42, 123, 456, 789, 1000)
    mode = modes[index // 20]
    case = index % 20
    return mode, lambdas[case // 5], seeds[case % 5]


def ablation_output_paths(project_root, mode, lam, seed):
    if mode not in ("original_control", "normalized"):
        raise ValueError("Unknown DDQN ablation mode")
    tag = os.environ.get("BRATU_DDQN_ABLATION_TAG", "production")
    if tag not in ("production", "smoke"):
        raise ValueError("Only production/smoke run tags are supported")
    directory = Path(project_root) / "results" / "ddqn_ablation_v1" / tag / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    stem = "ddqn_{}_lambda_{}_seed_{}".format(mode, str(lam).replace(".", "p"), seed)
    return directory / (stem + ".npz"), directory / (stem + ".json")


def save_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_ddqn_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(temp_path, str(path))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_npz_atomic(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_ddqn_", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(temp_path, str(path))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def source_fingerprint(project_root):
    """Hash the scientific code shared by both ablation treatments."""
    paths = (
        "experiments/bratu_ddqn_ablation.py",
        "experiments/bratu_ddqn_ablation_worker.py",
        "utils/ddqn_ablation_support.py",
        "utils/bratu_eval.py",
        "utils/fair_selection.py",
        "rl/ddqn_agent.py",
        "training/trainer.py",
        "problems/bratu.py",
        "models/pinn.py",
        "configs/bratu_config.py",
    )
    h = hashlib.sha256()
    for name in paths:
        h.update(name.encode("utf-8") + b"\0")
        with (Path(project_root) / name).open("rb") as fh:
            h.update(fh.read())
    return h.hexdigest()
