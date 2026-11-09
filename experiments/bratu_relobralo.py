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
    RELOBRALO_ALPHA, RELOBRALO_RHO, RELOBRALO_TEMPERATURE,
    WEIGHT_LOG_INTERVAL, SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE,
)
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from training.adaptive_weighting import ReLoBRaLoBalancer
from utils.bratu_eval import (
    evaluate_candidate_models, make_validation_grid, print_result_json,
    run_result_path, set_seed, validation_losses,
)
from utils.fair_selection import make_candidate_seeds, select_best_candidate
from utils.runtime_info import print_runtime_config

METHOD = "relobralo"
RELO_RNG_OFFSET = 7919


def train_candidate(problem, x_val, device, candidate_seed):
    set_seed(candidate_seed)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=HIDDEN_LAYERS).to(device)
    trainer = PINNTrainer(
        model=model, problem=problem, learning_rate=LEARNING_RATE,
        physics_weight=1.0, boundary_weight=1.0, device=device,
    )
    balancer = ReLoBRaLoBalancer(
        n_losses=2,
        alpha=RELOBRALO_ALPHA,
        rho_probability=RELOBRALO_RHO,
        temperature=RELOBRALO_TEMPERATURE,
        random_seed=int(candidate_seed) + RELO_RNG_OFFSET,
    )

    steps_log, alpha_log, beta_log, lf_log, lb_log = [], [], [], [], []
    rho_log, effective_alpha_log = [], []
    step_alpha_log, step_beta_log = [], []
    initial_alpha_log, initial_beta_log = [], []

    start = time.perf_counter()
    alpha_weight = 1.0
    beta_weight = 1.0

    for step in range(1, TOTAL_PINN_STEPS + 1):
        model.train()
        trainer.optimizer.zero_grad()
        _, physics_loss, boundary_loss = trainer.compute_loss(N_COLLOCATION)

        weights = balancer.update([
            float(physics_loss.detach().item()),
            float(boundary_loss.detach().item()),
        ])
        alpha_weight = float(weights[0])
        beta_weight = float(weights[1])

        total_loss = alpha_weight * physics_loss + beta_weight * boundary_loss
        total_loss.backward()
        trainer.optimizer.step()

        if step == 1 or step % WEIGHT_LOG_INTERVAL == 0 or step == TOTAL_PINN_STEPS:
            steps_log.append(step)
            alpha_log.append(alpha_weight)
            beta_log.append(beta_weight)
            lf_log.append(float(physics_loss.detach().item()))
            lb_log.append(float(boundary_loss.detach().item()))
            rho_log.append(float(balancer.last_rho))
            effective_alpha_log.append(float(balancer.last_effective_alpha))
            step_alpha_log.append(float(balancer.last_step_weights[0]))
            step_beta_log.append(float(balancer.last_step_weights[1]))
            initial_alpha_log.append(float(balancer.last_initial_weights[0]))
            initial_beta_log.append(float(balancer.last_initial_weights[1]))

    train_seconds = time.perf_counter() - start
    lf_val, lb_val = validation_losses(model, problem, x_val)
    return model, {
        "step": np.asarray(steps_log),
        "alpha": np.asarray(alpha_log),
        "beta": np.asarray(beta_log),
        "physics_loss": np.asarray(lf_log),
        "boundary_loss": np.asarray(lb_log),
        "rho": np.asarray(rho_log),
        "effective_alpha": np.asarray(effective_alpha_log),
        "step_alpha": np.asarray(step_alpha_log),
        "step_beta": np.asarray(step_beta_log),
        "initial_alpha": np.asarray(initial_alpha_log),
        "initial_beta": np.asarray(initial_beta_log),
        "final_alpha": float(alpha_weight),
        "final_beta": float(beta_weight),
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
            "ReLoBRaLo implementation": "Bischof-Kraus released reference update",
            "ReLoBRaLo alpha": RELOBRALO_ALPHA,
            "ReLoBRaLo rho P": RELOBRALO_RHO,
            "ReLoBRaLo temperature": RELOBRALO_TEMPERATURE,
            "ReLoBRaLo bootstrap": "alpha schedule 1, 0, then configured alpha",
            "ReLoBRaLo RNG": "per-candidate local RandomState",
        },
    )

    problem = BratuProblem(lam=LAMBDA)
    x_val = make_validation_grid(N_VALIDATION, device)
    models, infos, scores = [], [], []

    print("\n" + "=" * 78)
    print("ADAPTIVE-WEIGHTING BENCHMARK | RELOBRALO | MATCHED BEST-OF-{}".format(SELECTION_CANDIDATES))
    print("=" * 78)
    for idx, seed in enumerate(candidate_seeds, start=1):
        model, info = train_candidate(problem, x_val, device, seed)
        score = info["lf_val"] + info["lb_val"]
        models.append(model); infos.append(info); scores.append(score)
        print(
            "Candidate {:02d} | seed={} | S_val={:.6e} | final=(alpha={:.6e}, beta={:.6e}) | train={:.2f}s".format(
                idx, seed, score, info["final_alpha"], info["final_beta"], info["train_seconds"]
            )
        )

    best = select_best_candidate(scores)
    selected_candidate = best + 1
    selected_seed = candidate_seeds[best]
    selected_info = infos[best]
    validation_score = float(scores[best])

    # Reference/test information is introduced only after validation selection.
    candidate_results = evaluate_candidate_models(models, problem, N_TEST, device)
    result = candidate_results[best]

    output_file = run_result_path(PROJECT_ROOT, METHOD, LAMBDA, outer_seed)
    np.savez(
        output_file,
        protocol_version=np.asarray([PROTOCOL_VERSION]), selection_protocol=np.asarray([SELECTION_PROTOCOL]),
        method=np.asarray([METHOD]), x=result["x"], y_pred=result["y_pred"], y_ref=result["y_ref"],
        step=selected_info["step"], alpha=selected_info["alpha"], beta=selected_info["beta"],
        physics_loss=selected_info["physics_loss"], boundary_loss=selected_info["boundary_loss"],
        relobralo_rho=selected_info["rho"],
        relobralo_effective_alpha=selected_info["effective_alpha"],
        relobralo_step_alpha=selected_info["step_alpha"],
        relobralo_step_beta=selected_info["step_beta"],
        relobralo_initial_alpha=selected_info["initial_alpha"],
        relobralo_initial_beta=selected_info["initial_beta"],
        validation_score=np.asarray([validation_score]),
        candidate_seed=np.asarray(candidate_seeds, dtype=np.int64),
        candidate_physics_loss=np.asarray([x["lf_val"] for x in infos]),
        candidate_boundary_loss=np.asarray([x["lb_val"] for x in infos]),
        candidate_validation_score=np.asarray(scores),
        candidate_train_seconds=np.asarray([x["train_seconds"] for x in infos]),
        candidate_final_alpha=np.asarray([x["final_alpha"] for x in infos]),
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
    print("\nSelected candidate {} (seed {}) | S_val={:.6e} | relL2={:.6e} | final=({:.6e},{:.6e})".format(
        selected_candidate, selected_seed, validation_score, result["relative_l2"],
        selected_info["final_alpha"], selected_info["final_beta"]))
    print_result_json({
        "protocol_version": PROTOCOL_VERSION, "selection_protocol": SELECTION_PROTOCOL,
        "method": METHOD, "mse": result["mse"], "mae": result["mae"],
        "max_error": result["max_error"], "relative_l2": result["relative_l2"],
        "alpha": selected_info["final_alpha"], "beta": selected_info["final_beta"],
        "validation_score": validation_score,
        "selected_candidate": selected_candidate, "selected_candidate_seed": selected_seed,
        "n_candidates": SELECTION_CANDIDATES,
        "candidate_train_seconds_total": float(sum(times)),
        "candidate_train_seconds_mean": float(np.mean(times)),
    })
    print("Results saved to: {}".format(output_file))


if __name__ == "__main__":
    main()
