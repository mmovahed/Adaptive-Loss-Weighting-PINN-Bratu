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
    PROTOCOL_VERSION,
    SELECTION_PROTOCOL,
    LAMBDA,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    N_COLLOCATION,
    N_VALIDATION,
    N_TEST,
    TOTAL_PINN_STEPS,
    SELECTION_CANDIDATES,
    CANDIDATE_SEED_STRIDE,
)
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from utils.bratu_eval import (
    evaluate_candidate_models,
    make_validation_grid,
    print_result_json,
    run_result_path,
    set_seed,
    validation_losses,
)
from utils.fair_selection import make_candidate_seeds, select_best_candidate
from utils.runtime_info import print_runtime_config

METHOD = "fixed_equal"


def train_candidate(problem, x_val, device, candidate_seed):
    set_seed(candidate_seed)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=HIDDEN_LAYERS).to(device)
    trainer = PINNTrainer(
        model=model,
        problem=problem,
        learning_rate=LEARNING_RATE,
        physics_weight=1.0,
        boundary_weight=1.0,
        device=device,
    )

    start = time.perf_counter()
    history = trainer.train_adam(
        epochs=TOTAL_PINN_STEPS,
        n_collocation=N_COLLOCATION,
        print_every=max(1000, TOTAL_PINN_STEPS + 1),
    )
    train_seconds = time.perf_counter() - start

    lf_val, lb_val = validation_losses(model, problem, x_val)
    return model, history, lf_val, lb_val, train_seconds


def main():
    outer_seed = int(os.environ.get("BRATU_SEED", "42"))
    set_seed(outer_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_seeds = make_candidate_seeds(
        outer_seed, SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE
    )

    print_runtime_config(
        method=METHOD,
        outer_seed=outer_seed,
        candidate_seeds=candidate_seeds,
        device=device,
        extra={"Fixed weights": "alpha=1.0, beta=1.0"},
    )

    problem = BratuProblem(lam=LAMBDA)
    x_val = make_validation_grid(N_VALIDATION, device)

    candidate_models = []
    candidate_histories = []
    candidate_lf = []
    candidate_lb = []
    candidate_validation = []
    candidate_train_seconds = []

    print("\n" + "=" * 78)
    print("ADAPTIVE-WEIGHTING BENCHMARK | EQUAL WEIGHT | MATCHED BEST-OF-{}".format(SELECTION_CANDIDATES))
    print("=" * 78)

    for idx, candidate_seed in enumerate(candidate_seeds, start=1):
        model, history, lf_val, lb_val, seconds = train_candidate(
            problem, x_val, device, candidate_seed
        )
        score = lf_val + lb_val
        candidate_models.append(model)
        candidate_histories.append(history)
        candidate_lf.append(lf_val)
        candidate_lb.append(lb_val)
        candidate_validation.append(score)
        candidate_train_seconds.append(seconds)
        print(
            "Candidate {:02d} | seed={} | Lf_val={:.6e} | Lb_val={:.6e} | "
            "S_val={:.6e} | train={:.2f}s".format(
                idx, candidate_seed, lf_val, lb_val, score, seconds
            )
        )

    # Selection is validation-only. The reference solution has not been used yet.
    best_index = select_best_candidate(candidate_validation)
    selected_candidate = best_index + 1
    selected_seed = candidate_seeds[best_index]
    selected_history = candidate_histories[best_index]

    # Post-selection audit: evaluate every candidate against one common reference.
    candidate_results = evaluate_candidate_models(
        candidate_models, problem, N_TEST, device
    )
    result = candidate_results[best_index]

    candidate_rel = np.asarray([r["relative_l2"] for r in candidate_results])
    candidate_mse = np.asarray([r["mse"] for r in candidate_results])
    candidate_mae = np.asarray([r["mae"] for r in candidate_results])
    candidate_max = np.asarray([r["max_error"] for r in candidate_results])
    validation_score = float(candidate_validation[best_index])

    print("\n" + "=" * 78)
    print("EQUAL-WEIGHT PINN | SELECTED RESULT")
    print("=" * 78)
    print("Selected candidate : {}".format(selected_candidate))
    print("Selected seed      : {}".format(selected_seed))
    print("Validation score   : {:.6e}".format(validation_score))
    print("Relative L2        : {:.6e}".format(result["relative_l2"]))
    print("MSE                : {:.6e}".format(result["mse"]))
    print("Total train time   : {:.2f}s".format(sum(candidate_train_seconds)))
    print("=" * 78)

    output_file = run_result_path(PROJECT_ROOT, METHOD, LAMBDA, outer_seed)
    np.savez(
        output_file,
        protocol_version=np.asarray([PROTOCOL_VERSION]),
        selection_protocol=np.asarray([SELECTION_PROTOCOL]),
        method=np.asarray([METHOD]),
        x=result["x"], y_pred=result["y_pred"], y_ref=result["y_ref"],
        history_total=np.asarray(selected_history["total_loss"]),
        history_physics=np.asarray(selected_history["physics_loss"]),
        history_boundary=np.asarray(selected_history["boundary_loss"]),
        alpha=np.asarray([1.0]), beta=np.asarray([1.0]),
        validation_score=np.asarray([validation_score]),
        candidate_seed=np.asarray(candidate_seeds, dtype=np.int64),
        candidate_physics_loss=np.asarray(candidate_lf),
        candidate_boundary_loss=np.asarray(candidate_lb),
        candidate_validation_score=np.asarray(candidate_validation),
        candidate_train_seconds=np.asarray(candidate_train_seconds),
        candidate_test_relative_l2=candidate_rel,
        candidate_test_mse=candidate_mse,
        candidate_test_mae=candidate_mae,
        candidate_test_max_error=candidate_max,
        candidate_selected=(np.arange(SELECTION_CANDIDATES) == best_index).astype(np.int8),
        selected_candidate=np.asarray([selected_candidate], dtype=np.int64),
        selected_candidate_seed=np.asarray([selected_seed], dtype=np.int64),
    )

    print_result_json({
        "protocol_version": PROTOCOL_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "method": METHOD,
        "mse": result["mse"], "mae": result["mae"],
        "max_error": result["max_error"], "relative_l2": result["relative_l2"],
        "alpha": 1.0, "beta": 1.0,
        "validation_score": validation_score,
        "selected_candidate": selected_candidate,
        "selected_candidate_seed": selected_seed,
        "n_candidates": SELECTION_CANDIDATES,
        "candidate_train_seconds_total": float(sum(candidate_train_seconds)),
        "candidate_train_seconds_mean": float(np.mean(candidate_train_seconds)),
    })
    print("Results saved to: {}".format(output_file))


if __name__ == "__main__":
    main()
