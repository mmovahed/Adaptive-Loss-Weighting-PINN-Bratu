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
    LRA_SMOOTHING, LRA_UPDATE_INTERVAL, WEIGHT_LOG_INTERVAL,
    SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE,
)
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from training.adaptive_weighting import LearningRateAnnealingBalancer
from utils.bratu_eval import (
    evaluate_candidate_models, make_validation_grid, print_result_json,
    run_result_path, set_seed, validation_losses,
)
from utils.fair_selection import make_candidate_seeds, select_best_candidate
from utils.runtime_info import print_runtime_config

METHOD = "lra"


def train_candidate(problem, x_val, device, candidate_seed):
    set_seed(candidate_seed)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=HIDDEN_LAYERS).to(device)
    trainer = PINNTrainer(
        model=model, problem=problem, learning_rate=LEARNING_RATE,
        physics_weight=1.0, boundary_weight=1.0, device=device,
    )
    balancer = LearningRateAnnealingBalancer(
        smoothing=LRA_SMOOTHING,
        initial_boundary_weight=1.0,
    )

    steps_log, alpha_log, beta_log = [], [], []
    lf_log, lb_log, beta_hat_log = [], [], []
    numerator_log, denominator_log = [], []

    start = time.perf_counter()
    last_physics = None
    last_boundary = None

    for step in range(1, TOTAL_PINN_STEPS + 1):
        model.train()
        trainer.optimizer.zero_grad()
        _, physics_loss, boundary_loss = trainer.compute_loss(N_COLLOCATION)

        # Train with the coefficient available at the start of this iteration.
        total_loss = physics_loss + balancer.boundary_weight * boundary_loss
        total_loss.backward()
        trainer.optimizer.step()

        last_physics = float(physics_loss.detach().item())
        last_boundary = float(boundary_loss.detach().item())

        # Reference implementation refreshes the adaptive constant every
        # 10 optimizer iterations, after the optimizer update, and the new
        # value is therefore used by subsequent iterations.
        if step % LRA_UPDATE_INTERVAL == 0:
            trainer.optimizer.zero_grad()
            _, p_balance, b_balance = trainer.compute_loss(N_COLLOCATION)
            new_beta = balancer.update(p_balance, b_balance, model.parameters())
            trainer.boundary_weight = new_beta

        if step == 1 or step % WEIGHT_LOG_INTERVAL == 0 or step == TOTAL_PINN_STEPS:
            steps_log.append(step)
            alpha_log.append(1.0)
            beta_log.append(float(balancer.boundary_weight))
            lf_log.append(last_physics)
            lb_log.append(last_boundary)
            beta_hat_log.append(float(balancer.last_raw_weight))
            numerator_log.append(float(balancer.last_numerator))
            denominator_log.append(float(balancer.last_denominator))

    train_seconds = time.perf_counter() - start
    lf_val, lb_val = validation_losses(model, problem, x_val)
    return model, {
        "step": np.asarray(steps_log),
        "alpha": np.asarray(alpha_log),
        "beta": np.asarray(beta_log),
        "physics_loss": np.asarray(lf_log),
        "boundary_loss": np.asarray(lb_log),
        "beta_hat": np.asarray(beta_hat_log),
        "gradient_numerator": np.asarray(numerator_log),
        "gradient_denominator": np.asarray(denominator_log),
        "final_beta": float(balancer.boundary_weight),
        "n_weight_updates": int(balancer.n_updates),
        "lf_val": float(lf_val),
        "lb_val": float(lb_val),
        "train_seconds": float(train_seconds),
    }


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
            "LRA implementation": "Wang et al. released reference code",
            "LRA smoothing": LRA_SMOOTHING,
            "LRA update interval": LRA_UPDATE_INTERVAL,
            "LRA gradient params": "network weight tensors (biases excluded)",
            "LRA denominator": "grad(current_beta * boundary_loss)",
        },
    )

    problem = BratuProblem(lam=LAMBDA)
    x_val = make_validation_grid(N_VALIDATION, device)
    models, infos, scores = [], [], []

    print("\n" + "=" * 78)
    print("ADAPTIVE-WEIGHTING BENCHMARK | LRA | MATCHED BEST-OF-{}".format(SELECTION_CANDIDATES))
    print("=" * 78)
    for idx, seed in enumerate(candidate_seeds, start=1):
        model, info = train_candidate(problem, x_val, device, seed)
        score = info["lf_val"] + info["lb_val"]
        models.append(model); infos.append(info); scores.append(score)
        print(
            "Candidate {:02d} | seed={} | S_val={:.6e} | final_beta={:.6e} | "
            "LRA_updates={} | train={:.2f}s".format(
                idx, seed, score, info["final_beta"], info["n_weight_updates"],
                info["train_seconds"]
            )
        )

    best = select_best_candidate(scores)
    selected_candidate = best + 1
    selected_seed = candidate_seeds[best]
    selected_info = infos[best]
    validation_score = float(scores[best])

    candidate_results = evaluate_candidate_models(models, problem, N_TEST, device)
    result = candidate_results[best]

    output_file = run_result_path(PROJECT_ROOT, METHOD, LAMBDA, outer_seed)
    np.savez(
        output_file,
        protocol_version=np.asarray([PROTOCOL_VERSION]), selection_protocol=np.asarray([SELECTION_PROTOCOL]),
        method=np.asarray([METHOD]), x=result["x"], y_pred=result["y_pred"], y_ref=result["y_ref"],
        step=selected_info["step"], alpha=selected_info["alpha"], beta=selected_info["beta"],
        physics_loss=selected_info["physics_loss"], boundary_loss=selected_info["boundary_loss"],
        lra_beta_hat=selected_info["beta_hat"],
        lra_gradient_numerator=selected_info["gradient_numerator"],
        lra_gradient_denominator=selected_info["gradient_denominator"],
        validation_score=np.asarray([validation_score]),
        candidate_seed=np.asarray(candidate_seeds, dtype=np.int64),
        candidate_physics_loss=np.asarray([x["lf_val"] for x in infos]),
        candidate_boundary_loss=np.asarray([x["lb_val"] for x in infos]),
        candidate_validation_score=np.asarray(scores),
        candidate_train_seconds=np.asarray([x["train_seconds"] for x in infos]),
        candidate_final_alpha=np.ones(SELECTION_CANDIDATES),
        candidate_final_beta=np.asarray([x["final_beta"] for x in infos]),
        candidate_test_relative_l2=np.asarray([r["relative_l2"] for r in candidate_results]),
        candidate_test_mse=np.asarray([r["mse"] for r in candidate_results]),
        candidate_test_mae=np.asarray([r["mae"] for r in candidate_results]),
        candidate_test_max_error=np.asarray([r["max_error"] for r in candidate_results]),
        candidate_selected=(np.arange(SELECTION_CANDIDATES) == best).astype(np.int8),
        selected_candidate=np.asarray([selected_candidate], dtype=np.int64),
        selected_candidate_seed=np.asarray([selected_seed], dtype=np.int64),
    )

    times = [x["train_seconds"] for x in infos]
    print("\nSelected candidate {} (seed {}) | S_val={:.6e} | relL2={:.6e} | final_beta={:.6e}".format(
        selected_candidate, selected_seed, validation_score, result["relative_l2"], selected_info["final_beta"]))
    print_result_json({
        "protocol_version": PROTOCOL_VERSION, "selection_protocol": SELECTION_PROTOCOL,
        "method": METHOD, "mse": result["mse"], "mae": result["mae"],
        "max_error": result["max_error"], "relative_l2": result["relative_l2"],
        "alpha": 1.0, "beta": selected_info["final_beta"],
        "validation_score": validation_score,
        "selected_candidate": selected_candidate, "selected_candidate_seed": selected_seed,
        "n_candidates": SELECTION_CANDIDATES,
        "candidate_train_seconds_total": float(sum(times)),
        "candidate_train_seconds_mean": float(np.mean(times)),
    })
    print("Results saved to: {}".format(output_file))


if __name__ == "__main__":
    main()
