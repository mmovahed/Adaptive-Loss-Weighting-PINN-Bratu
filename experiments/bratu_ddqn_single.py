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
    WARMUP_STEPS, DECISION_INTERVAL, DDQN_TRAIN_EPISODES,
    INITIAL_ALPHA, INITIAL_BETA, MIN_WEIGHT, MAX_WEIGHT,
    SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE,
)
from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from rl.ddqn_agent import DDQNAgent
from utils.bratu_eval import (
    evaluate_candidate_models, make_validation_grid, print_result_json,
    run_result_path, set_seed, validation_losses,
)
from utils.fair_selection import make_candidate_seeds, select_best_candidate
from utils.runtime_info import print_runtime_config

METHOD = "ddqn"
MULTIPLIERS = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)


def build_state(
    physics_loss, boundary_loss, previous_physics_loss, previous_boundary_loss,
    physics_weight, boundary_weight, step, total_steps,
):
    eps = 1e-12
    log_physics = np.log(physics_loss + eps)
    log_boundary = np.log(boundary_loss + eps)
    return np.asarray([
        log_physics,
        log_boundary,
        log_physics - np.log(previous_physics_loss + eps),
        log_boundary - np.log(previous_boundary_loss + eps),
        physics_weight / MAX_WEIGHT,
        boundary_weight / MAX_WEIGHT,
        step / max(total_steps, 1),
    ], dtype=np.float32)


def apply_joint_action(alpha, beta, action):
    if action < 0 or action >= 25:
        raise ValueError("Joint action must be in [0, 24], got {}.".format(action))
    alpha_multiplier = MULTIPLIERS[action // 5]
    beta_multiplier = MULTIPLIERS[action % 5]
    new_alpha = np.clip(alpha * alpha_multiplier, MIN_WEIGHT, MAX_WEIGHT)
    new_beta = np.clip(beta * beta_multiplier, MIN_WEIGHT, MAX_WEIGHT)
    return float(new_alpha), float(new_beta)


def train_steps(trainer, n_steps):
    for _ in range(n_steps):
        trainer.train_adam(epochs=1, n_collocation=N_COLLOCATION, print_every=10 ** 9)


def run_controller_episode(agent, problem, x_val, device, episode_seed):
    """Train one candidate PINN while the persistent DDQN learns online.

    No numerical reference solution is accessed here. Controller/replay state is
    intentionally persistent across the five candidate PINNs; those five PINN
    trajectories are the DDQN training budget and are also the candidate pool.
    """
    set_seed(episode_seed)
    model = PINN(input_dim=1, output_dim=1, hidden_layers=HIDDEN_LAYERS).to(device)

    alpha = INITIAL_ALPHA
    beta = INITIAL_BETA
    trainer = PINNTrainer(
        model=model, problem=problem, learning_rate=LEARNING_RATE,
        physics_weight=alpha, boundary_weight=beta, device=device,
    )

    warmup = min(WARMUP_STEPS, TOTAL_PINN_STEPS)
    train_steps(trainer, warmup)
    physics_loss, boundary_loss = validation_losses(model, problem, x_val)
    state = build_state(
        physics_loss, boundary_loss, physics_loss, boundary_loss,
        alpha, beta, warmup, TOTAL_PINN_STEPS,
    )

    trajectory_step = [warmup]
    trajectory_alpha = [alpha]
    trajectory_beta = [beta]
    trajectory_lf = [physics_loss]
    trajectory_lb = [boundary_loss]
    trajectory_score = [physics_loss + boundary_loss]
    trajectory_action = [-1]
    trajectory_reward = [0.0]
    trajectory_q_loss = [np.nan]
    episode_reward = 0.0

    for decision_start in range(warmup + 1, TOTAL_PINN_STEPS + 1, DECISION_INTERVAL):
        action = agent.select_action(state)
        alpha, beta = apply_joint_action(alpha, beta, action)
        trainer.physics_weight = alpha
        trainer.boundary_weight = beta

        score_before = physics_loss + boundary_loss
        old_physics = physics_loss
        old_boundary = boundary_loss
        steps_this_interval = min(
            DECISION_INTERVAL, TOTAL_PINN_STEPS - decision_start + 1
        )
        train_steps(trainer, steps_this_interval)

        physics_loss, boundary_loss = validation_losses(model, problem, x_val)
        score_after = physics_loss + boundary_loss
        reward = float(np.tanh(np.log(
            (score_before + 1e-12) / (score_after + 1e-12)
        )))
        episode_reward += reward

        next_step = decision_start + steps_this_interval - 1
        next_state = build_state(
            physics_loss, boundary_loss, old_physics, old_boundary,
            alpha, beta, next_step, TOTAL_PINN_STEPS,
        )
        done = next_step >= TOTAL_PINN_STEPS
        agent.remember(state, action, reward, next_state, done)
        q_loss = agent.train_step()

        trajectory_step.append(next_step)
        trajectory_alpha.append(alpha)
        trajectory_beta.append(beta)
        trajectory_lf.append(physics_loss)
        trajectory_lb.append(boundary_loss)
        trajectory_score.append(score_after)
        trajectory_action.append(action)
        trajectory_reward.append(reward)
        trajectory_q_loss.append(np.nan if q_loss is None else q_loss)
        state = next_state

    return model, {
        "reward": float(episode_reward),
        "lf_val": float(physics_loss),
        "lb_val": float(boundary_loss),
        "validation_score": float(physics_loss + boundary_loss),
        "alpha": float(alpha), "beta": float(beta),
        "step": np.asarray(trajectory_step),
        "alpha_trajectory": np.asarray(trajectory_alpha),
        "beta_trajectory": np.asarray(trajectory_beta),
        "physics_loss": np.asarray(trajectory_lf),
        "boundary_loss": np.asarray(trajectory_lb),
        "validation_score_trajectory": np.asarray(trajectory_score),
        "action": np.asarray(trajectory_action),
        "reward_trajectory": np.asarray(trajectory_reward),
        "q_loss": np.asarray(trajectory_q_loss),
    }


def main():
    outer_seed = int(os.environ.get("BRATU_SEED", "42"))
    set_seed(outer_seed)
    if DDQN_TRAIN_EPISODES != SELECTION_CANDIDATES:
        raise ValueError(
            "DDQN candidate pool and training episodes must match: {} != {}".format(
                DDQN_TRAIN_EPISODES, SELECTION_CANDIDATES
            )
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_seeds = make_candidate_seeds(
        outer_seed, SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE
    )
    print_runtime_config(
        method=METHOD, outer_seed=outer_seed, candidate_seeds=candidate_seeds,
        device=device,
        extra={
            "DDQN state dim": 7, "DDQN action dim": 25,
            "DDQN replay size": 5000, "DDQN batch size": 32,
            "DDQN target update": 100,
            "DDQN protocol": "persistent controller/replay across candidate PINNs",
        },
    )

    problem = BratuProblem(lam=LAMBDA)
    x_val = make_validation_grid(N_VALIDATION, device)

    # Outer seed fixes the initial Q-networks. Candidate seeds fix PINN
    # initializations and stochastic actions within each candidate trajectory.
    set_seed(outer_seed)
    agent = DDQNAgent(
        state_dim=7, action_dim=25, learning_rate=1e-3, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        batch_size=32, memory_size=5000, target_update_frequency=100,
        device=device,
    )

    models, infos, scores, epsilons, times = [], [], [], [], []
    print("\n" + "=" * 78)
    print("ADAPTIVE-WEIGHTING BENCHMARK | DDQN | MATCHED BEST-OF-{}".format(SELECTION_CANDIDATES))
    print("Controller/replay persist across candidates; no fresh sixth PINN.")
    print("=" * 78)

    for idx, seed in enumerate(candidate_seeds, start=1):
        start = time.perf_counter()
        model, info = run_controller_episode(agent, problem, x_val, device, seed)
        seconds = time.perf_counter() - start
        info["train_seconds"] = float(seconds)
        models.append(model); infos.append(info); scores.append(info["validation_score"])
        epsilons.append(float(agent.epsilon)); times.append(float(seconds))
        print(
            "Candidate {:02d} | seed={} | S_val={:.6e} | Reward={:.6e} | "
            "final=({:.3e},{:.3e}) | eps={:.4f} | train={:.2f}s".format(
                idx, seed, info["validation_score"], info["reward"], info["alpha"],
                info["beta"], agent.epsilon, seconds
            )
        )

    best = select_best_candidate(scores)
    selected_candidate = best + 1
    selected_seed = candidate_seeds[best]
    selected_info = infos[best]
    validation_score = float(scores[best])

    # Test/reference evaluation starts only after the validation selector is frozen.
    candidate_results = evaluate_candidate_models(models, problem, N_TEST, device)
    result = candidate_results[best]

    output_file = run_result_path(PROJECT_ROOT, METHOD, LAMBDA, outer_seed)
    np.savez(
        output_file,
        protocol_version=np.asarray([PROTOCOL_VERSION]), selection_protocol=np.asarray([SELECTION_PROTOCOL]),
        method=np.asarray([METHOD]), x=result["x"], y_pred=result["y_pred"], y_ref=result["y_ref"],
        step=selected_info["step"], alpha=selected_info["alpha_trajectory"],
        beta=selected_info["beta_trajectory"], physics_loss=selected_info["physics_loss"],
        boundary_loss=selected_info["boundary_loss"],
        validation_score_trajectory=selected_info["validation_score_trajectory"],
        action=selected_info["action"], reward=selected_info["reward_trajectory"],
        q_loss=selected_info["q_loss"], validation_score=np.asarray([validation_score]),
        candidate_seed=np.asarray(candidate_seeds, dtype=np.int64),
        candidate_physics_loss=np.asarray([x["lf_val"] for x in infos]),
        candidate_boundary_loss=np.asarray([x["lb_val"] for x in infos]),
        candidate_reward=np.asarray([x["reward"] for x in infos]),
        candidate_validation_score=np.asarray(scores),
        candidate_terminal_alpha=np.asarray([x["alpha"] for x in infos]),
        candidate_terminal_beta=np.asarray([x["beta"] for x in infos]),
        candidate_epsilon=np.asarray(epsilons),
        candidate_train_seconds=np.asarray(times),
        candidate_test_relative_l2=np.asarray([r["relative_l2"] for r in candidate_results]),
        candidate_test_mse=np.asarray([r["mse"] for r in candidate_results]),
        candidate_test_mae=np.asarray([r["mae"] for r in candidate_results]),
        candidate_test_max_error=np.asarray([r["max_error"] for r in candidate_results]),
        candidate_selected=(np.arange(SELECTION_CANDIDATES) == best).astype(np.int8),
        selected_candidate=np.asarray([selected_candidate], dtype=np.int64),
        selected_candidate_seed=np.asarray([selected_seed], dtype=np.int64),
    )

    print("\nSelected candidate {} (seed {}) | S_val={:.6e} | relL2={:.6e} | final=({:.6e},{:.6e})".format(
        selected_candidate, selected_seed, validation_score, result["relative_l2"],
        selected_info["alpha"], selected_info["beta"]))
    print_result_json({
        "protocol_version": PROTOCOL_VERSION, "selection_protocol": SELECTION_PROTOCOL,
        "method": METHOD, "mse": result["mse"], "mae": result["mae"],
        "max_error": result["max_error"], "relative_l2": result["relative_l2"],
        "alpha": selected_info["alpha"], "beta": selected_info["beta"],
        "validation_score": validation_score,
        "selected_candidate": selected_candidate, "selected_candidate_seed": selected_seed,
        "n_candidates": SELECTION_CANDIDATES,
        "selected_episode_reward": selected_info["reward"],
        "candidate_train_seconds_total": float(sum(times)),
        "candidate_train_seconds_mean": float(np.mean(times)),
    })
    print("Results saved to: {}".format(output_file))


if __name__ == "__main__":
    main()
