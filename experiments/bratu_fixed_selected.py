import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.bratu_config import (
    PROTOCOL_VERSION, SELECTION_PROTOCOL, LAMBDA, HIDDEN_LAYERS, LEARNING_RATE,
    N_COLLOCATION, N_VALIDATION, N_TEST, TOTAL_PINN_STEPS,
    SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE,
    FIXED_SELECTED_ALPHA, FIXED_SELECTED_BETA,
)
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from utils.bratu_eval import (
    evaluate_candidate_models, make_validation_grid, print_result_json,
    run_result_path, set_seed, validation_losses,
)
from utils.fair_selection import make_candidate_seeds, select_best_candidate
from utils.runtime_info import print_runtime_config

METHOD = "fixed_selected"
ALPHA = FIXED_SELECTED_ALPHA
BETA = FIXED_SELECTED_BETA


def train_candidate(problem, x_val, device, candidate_seed):
    set_seed(candidate_seed)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=HIDDEN_LAYERS).to(device)
    trainer = PINNTrainer(
        model=model, problem=problem, learning_rate=LEARNING_RATE,
        physics_weight=ALPHA, boundary_weight=BETA, device=device,
    )
    start = time.perf_counter()
    history = trainer.train_adam(
        epochs=TOTAL_PINN_STEPS,
        n_collocation=N_COLLOCATION,
        print_every=max(1000, TOTAL_PINN_STEPS + 1),
    )
    seconds = time.perf_counter() - start
    lf_val, lb_val = validation_losses(model, problem, x_val)
    return model, history, lf_val, lb_val, seconds


def main():
    outer_seed = int(os.environ.get("BRATU_SEED", "42"))
    set_seed(outer_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_seeds = make_candidate_seeds(
        outer_seed, SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE
    )
    print_runtime_config(
        method=METHOD, outer_seed=outer_seed, candidate_seeds=candidate_seeds,
        device=device,
        extra={
            "Fixed weights": "alpha={}, beta={}".format(ALPHA, BETA),
            "Role": "locked contextual static comparator; no benchmark-v2 tuning",
        },
    )

    problem = BratuProblem(lam=LAMBDA)
    x_val = make_validation_grid(N_VALIDATION, device)
    models, histories, lf_vals, lb_vals, scores, times = [], [], [], [], [], []

    print("\n" + "=" * 78)
    print("ADAPTIVE-WEIGHTING BENCHMARK | FIXED ({},{}) | MATCHED BEST-OF-{}".format(ALPHA, BETA, SELECTION_CANDIDATES))
    print("=" * 78)
    for idx, seed in enumerate(candidate_seeds, start=1):
        model, history, lf, lb, seconds = train_candidate(problem, x_val, device, seed)
        score = lf + lb
        models.append(model); histories.append(history); lf_vals.append(lf); lb_vals.append(lb)
        scores.append(score); times.append(seconds)
        print("Candidate {:02d} | seed={} | Lf_val={:.6e} | Lb_val={:.6e} | S_val={:.6e} | train={:.2f}s".format(
            idx, seed, lf, lb, score, seconds))

    best = select_best_candidate(scores)
    selected_candidate = best + 1
    selected_seed = candidate_seeds[best]
    candidate_results = evaluate_candidate_models(models, problem, N_TEST, device)
    result = candidate_results[best]
    validation_score = float(scores[best])

    output_file = run_result_path(PROJECT_ROOT, METHOD, LAMBDA, outer_seed)
    np.savez(
        output_file,
        protocol_version=np.asarray([PROTOCOL_VERSION]), selection_protocol=np.asarray([SELECTION_PROTOCOL]),
        method=np.asarray([METHOD]), x=result["x"], y_pred=result["y_pred"], y_ref=result["y_ref"],
        history_total=np.asarray(histories[best]["total_loss"]),
        history_physics=np.asarray(histories[best]["physics_loss"]),
        history_boundary=np.asarray(histories[best]["boundary_loss"]),
        alpha=np.asarray([ALPHA]), beta=np.asarray([BETA]), validation_score=np.asarray([validation_score]),
        candidate_seed=np.asarray(candidate_seeds, dtype=np.int64),
        candidate_physics_loss=np.asarray(lf_vals), candidate_boundary_loss=np.asarray(lb_vals),
        candidate_validation_score=np.asarray(scores), candidate_train_seconds=np.asarray(times),
        candidate_test_relative_l2=np.asarray([r["relative_l2"] for r in candidate_results]),
        candidate_test_mse=np.asarray([r["mse"] for r in candidate_results]),
        candidate_test_mae=np.asarray([r["mae"] for r in candidate_results]),
        candidate_test_max_error=np.asarray([r["max_error"] for r in candidate_results]),
        candidate_selected=(np.arange(SELECTION_CANDIDATES) == best).astype(np.int8),
        selected_candidate=np.asarray([selected_candidate], dtype=np.int64),
        selected_candidate_seed=np.asarray([selected_seed], dtype=np.int64),
    )

    print("\nSelected candidate {} (seed {}) | S_val={:.6e} | relL2={:.6e}".format(
        selected_candidate, selected_seed, validation_score, result["relative_l2"]))
    print_result_json({
        "protocol_version": PROTOCOL_VERSION, "selection_protocol": SELECTION_PROTOCOL,
        "method": METHOD, "mse": result["mse"], "mae": result["mae"],
        "max_error": result["max_error"], "relative_l2": result["relative_l2"],
        "alpha": ALPHA, "beta": BETA, "validation_score": validation_score,
        "selected_candidate": selected_candidate, "selected_candidate_seed": selected_seed,
        "n_candidates": SELECTION_CANDIDATES,
        "candidate_train_seconds_total": float(sum(times)),
        "candidate_train_seconds_mean": float(np.mean(times)),
    })
    print("Results saved to: {}".format(output_file))


if __name__ == "__main__":
    main()
