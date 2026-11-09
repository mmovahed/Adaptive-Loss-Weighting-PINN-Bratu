import sys
from pathlib import Path
import random
import os

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from rl.ddqn_agent import DDQNAgent

from configs.bratu_config import (
    LAMBDA,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    N_COLLOCATION,
    N_VALIDATION,
    N_TEST,
    TOTAL_PINN_STEPS,
    WARMUP_STEPS,
    DECISION_INTERVAL,
    INITIAL_ALPHA,
    INITIAL_BETA,
    MIN_WEIGHT,
    MAX_WEIGHT,
)

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Validation losses
# ============================================================

def validation_losses(model, problem, x_val):

    model.eval()

    x = (
        x_val
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # --------------------------------------------------------
    # Model output
    # --------------------------------------------------------

    u = model(x)

    # --------------------------------------------------------
    # First derivative
    # --------------------------------------------------------

    du = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    # --------------------------------------------------------
    # Second derivative
    # --------------------------------------------------------

    d2u = torch.autograd.grad(
        du,
        x,
        grad_outputs=torch.ones_like(du),
        create_graph=True,
    )[0]

    # --------------------------------------------------------
    # Bratu residual
    #
    # u''(x) + lambda * exp(u(x)) = 0
    # --------------------------------------------------------

    residual = (
        d2u
        + problem.lam * torch.exp(u)
    )

    physics_loss = torch.mean(
        residual ** 2
    )

    # --------------------------------------------------------
    # Boundary loss
    #
    # u(0) = 0
    # u(1) = 0
    # --------------------------------------------------------

    x_boundary = torch.tensor(
        [[0.0], [1.0]],
        dtype=torch.float32,
        device=x_val.device,
    )

    with torch.no_grad():
        u_boundary = model(
            x_boundary
        )

    boundary_loss = torch.mean(
        u_boundary ** 2
    )

    return (
        float(physics_loss.item()),
        float(boundary_loss.item()),
    )


# ============================================================
# State
# ============================================================

def build_state(
    physics_loss,
    boundary_loss,
    previous_physics_loss,
    previous_boundary_loss,
    physics_weight,
    boundary_weight,
    step,
    total_steps,
):

    eps = 1e-12

    log_physics = np.log(
        physics_loss + eps
    )

    log_boundary = np.log(
        boundary_loss + eps
    )

    delta_log_physics = (
        log_physics
        - np.log(
            previous_physics_loss + eps
        )
    )

    delta_log_boundary = (
        log_boundary
        - np.log(
            previous_boundary_loss + eps
        )
    )

    state = np.array(
        [
            log_physics,
            log_boundary,
            delta_log_physics,
            delta_log_boundary,
            physics_weight / 100.0,
            boundary_weight / 100.0,
            step / total_steps,
        ],
        dtype=np.float32,
    )

    return state


# ============================================================
# Apply action
# ============================================================

def apply_action(
    weight,
    action,
    min_weight=MIN_WEIGHT,
    max_weight=MAX_WEIGHT,
):

    multipliers = np.array(
        [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
        ],
        dtype=np.float32,
    )

    new_weight = (
        weight
        * multipliers[action]
    )

    new_weight = np.clip(
        new_weight,
        min_weight,
        max_weight,
    )

    return float(new_weight)


# ============================================================
# Final evaluation
# ============================================================

def evaluate_final(
    model,
    problem,
    device,
):

    x_test = torch.linspace(
        0.0,
        1.0,
        N_TEST,
        device=device,
    ).reshape(-1, 1)

    model.eval()

    with torch.no_grad():
        y_pred = model(
            x_test
        )

    # --------------------------------------------------------
    # Reference solution is used ONLY for final evaluation
    # --------------------------------------------------------

    _, y_ref = problem.reference_solution(
        n_points=N_TEST
    )

    y_ref_tensor = torch.tensor(
        y_ref,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    error = (
        y_pred
        - y_ref_tensor
    )

    mse = torch.mean(
        error ** 2
    ).item()

    mae = torch.mean(
        torch.abs(error)
    ).item()

    max_error = torch.max(
        torch.abs(error)
    ).item()

    relative_l2 = (
        torch.norm(error)
        /
        (
            torch.norm(
                y_ref_tensor
            )
            + 1e-12
        )
    ).item()

    return (
        mse,
        mae,
        max_error,
        relative_l2,
    )


# ============================================================
# Main
# ============================================================

def main():

    seed = int(
        os.environ.get(
            "BRATU_SEED",
            "42",
            )
        )

    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Bratu problem
    # ========================================================

    problem = BratuProblem(
        lam=LAMBDA
    )
    # ========================================================
    # Fixed validation set
    # ========================================================

    x_val = torch.linspace(
        0.0,
        1.0,
        N_VALIDATION,
        device=device,
    ).reshape(-1, 1)

    # ========================================================
    # Two independent DDQN agents
    # ========================================================

    agent_alpha = DDQNAgent(
        state_dim=7,
        action_dim=5,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=5000,
        target_update_frequency=100,
        device=device,
    )

    agent_beta = DDQNAgent(
        state_dim=7,
        action_dim=5,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=5000,
        target_update_frequency=100,
        device=device,
    )

    # ========================================================
    # Training settings
    # ========================================================

    episodes = 5

    warmup_steps = WARMUP_STEPS
    total_steps = TOTAL_PINN_STEPS
    decision_interval = DECISION_INTERVAL

    # ========================================================
    # Result storage
    # ========================================================

    episode_numbers = []
    episode_rewards = []

    episode_mse = []
    episode_mae = []
    episode_max_error = []
    episode_relative_l2 = []

    episode_physics_weights = []
    episode_boundary_weights = []

    episode_physics_loss = []
    episode_boundary_loss = []
    episode_validation_score = []

    episode_alpha_epsilon = []
    episode_beta_epsilon = []

    # ========================================================
    # Training
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BRATU TWO-AGENT DDQN TRAINING"
    )
    print("=" * 70)

    for episode in range(
        1,
        episodes + 1,
    ):

        # ====================================================
        # Fresh PINN
        # ====================================================

        model = PINN(
            input_dim=1,
            output_dim=1,
            hidden_layers=HIDDEN_LAYERS,
        ).to(device)

        # ----------------------------------------------------
        # Initial weights
        # ----------------------------------------------------

        physics_weight = INITIAL_ALPHA
        boundary_weight = INITIAL_BETA

        # ----------------------------------------------------
        # One persistent Trainer / Adam optimizer
        # ----------------------------------------------------

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            learning_rate=1e-3,
            physics_weight=physics_weight,
            boundary_weight=boundary_weight,
            device=device,
        )

        # ====================================================
        # Initial validation
        # ====================================================

        physics_loss, boundary_loss = (
            validation_losses(
                model,
                problem,
                x_val,
            )
        )

        previous_physics_loss = (
            physics_loss
        )

        previous_boundary_loss = (
            boundary_loss
        )

        # ====================================================
        # Warm-up
        # ====================================================

        for step in range(
            1,
            warmup_steps + 1,
        ):

            trainer.train_adam(
                epochs=1,
                n_collocation=N_COLLOCATION,
                print_every=999999,
            )

        # ----------------------------------------------------
        # Validation after warm-up
        # ----------------------------------------------------

        physics_loss, boundary_loss = (
            validation_losses(
                model,
                problem,
                x_val,
            )
        )

        previous_physics_loss = (
            physics_loss
        )

        previous_boundary_loss = (
            boundary_loss
        )

        # ====================================================
        # RL phase
        # ====================================================

        episode_reward = 0.0

        # Current decision state
        state = build_state(
            physics_loss,
            boundary_loss,
            previous_physics_loss,
            previous_boundary_loss,
            physics_weight,
            boundary_weight,
            warmup_steps,
            total_steps,
        )

        # ----------------------------------------------------
        # Number of decisions
        # ----------------------------------------------------

        first_rl_step = (
            warmup_steps + 1
        )

        for decision_start in range(
            first_rl_step,
            total_steps + 1,
            decision_interval,
        ):

            # =================================================
            # Agents select actions
            # =================================================

            action_alpha = (
                agent_alpha.select_action(
                    state
                )
            )

            action_beta = (
                agent_beta.select_action(
                    state
                )
            )

            # -------------------------------------------------
            # Update alpha and beta
            # -------------------------------------------------

            physics_weight = apply_action(
                physics_weight,
                action_alpha,
            )

            boundary_weight = apply_action(
                boundary_weight,
                action_beta,
            )

            # -------------------------------------------------
            # Update trainer weights
            # -------------------------------------------------

            trainer.physics_weight = (
                physics_weight
            )

            trainer.boundary_weight = (
                boundary_weight
            )

            # =================================================
            # Save validation score BEFORE this decision
            # =================================================

            score_before = (
                physics_loss
                + boundary_loss
            )

            old_physics_loss = (
                physics_loss
            )

            old_boundary_loss = (
                boundary_loss
            )

            # =================================================
            # Train PINN for one decision interval
            # =================================================

            steps_this_interval = min(
                decision_interval,
                total_steps
                - decision_start
                + 1,
            )

            for _ in range(
                steps_this_interval
            ):

                trainer.train_adam(
                    epochs=1,
                    n_collocation=N_COLLOCATION,
                    print_every=999999,
                )

            # =================================================
            # Validation AFTER interval
            # =================================================

            physics_loss, boundary_loss = (
                validation_losses(
                    model,
                    problem,
                    x_val,
                )
            )

            score_after = (
                physics_loss
                + boundary_loss
            )

            # =================================================
            # Shared logarithmic reward
            # =================================================

            reward = np.tanh(
                np.log(
                    (score_before + 1e-12)
                    /
                    (score_after + 1e-12)
                )
            )

            episode_reward += (
                reward
            )

            # =================================================
            # Next state
            # =================================================

            next_step = (
                decision_start
                + steps_this_interval
                - 1
            )

            next_state = build_state(
                physics_loss,
                boundary_loss,
                old_physics_loss,
                old_boundary_loss,
                physics_weight,
                boundary_weight,
                next_step,
                total_steps,
            )

            done = (
                next_step >= total_steps
            )

            # =================================================
            # Replay transition for both agents
            # =================================================

            agent_alpha.remember(
                state,
                action_alpha,
                reward,
                next_state,
                done,
            )

            agent_beta.remember(
                state,
                action_beta,
                reward,
                next_state,
                done,
            )

            # =================================================
            # Train both DDQN agents
            # =================================================

            agent_alpha.train_step()
            agent_beta.train_step()

            # =================================================
            # Update state
            # =================================================

            state = next_state

            previous_physics_loss = (
                old_physics_loss
            )

            previous_boundary_loss = (
                old_boundary_loss
            )

        # ====================================================
        # Final evaluation
        # ====================================================

        (
            mse,
            mae,
            max_error,
            relative_l2,
        ) = evaluate_final(
            model,
            problem,
            device,
        )

        # ====================================================
        # Save episode results
        # ====================================================

        episode_numbers.append(
            episode
        )

        episode_rewards.append(
            episode_reward
        )

        episode_mse.append(
            mse
        )

        episode_mae.append(
            mae
        )

        episode_max_error.append(
            max_error
        )

        episode_relative_l2.append(
            relative_l2
        )

        episode_physics_weights.append(
            physics_weight
        )

        episode_boundary_weights.append(
            boundary_weight
        )

        episode_physics_loss.append(
            physics_loss
        )

        episode_boundary_loss.append(
            boundary_loss
        )

        episode_validation_score.append(
            physics_loss + boundary_loss
        )

        episode_alpha_epsilon.append(
            agent_alpha.epsilon
        )

        episode_beta_epsilon.append(
            agent_beta.epsilon
        )

        # ====================================================
        # Print episode result
        # ====================================================

        print(
            f"\nEpisode {episode:02d} | "
            f"Reward: "
            f"{episode_reward:.6e} | "
            f"MSE: "
            f"{mse:.6e} | "
            f"RelL2: "
            f"{relative_l2:.6e} | "
            f"ValScore: "
            f"{(physics_loss + boundary_loss):.6e} | "
            f"Alpha: "
            f"{physics_weight:.4f} | "
            f"Beta: "
            f"{boundary_weight:.4f} | "
            f"Eps-A: "
            f"{agent_alpha.epsilon:.4f} | "
            f"Eps-B: "
            f"{agent_beta.epsilon:.4f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    results_dir = (
        PROJECT_ROOT / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        results_dir
        / "bratu_two_agent_ddqn_results.npz"
    )

    np.savez(
        output_file,

        episode=np.array(
            episode_numbers
        ),

        reward=np.array(
            episode_rewards
        ),

        mse=np.array(
            episode_mse
        ),

        mae=np.array(
            episode_mae
        ),

        max_error=np.array(
            episode_max_error
        ),

        relative_l2=np.array(
            episode_relative_l2
        ),

        physics_weight=np.array(
            episode_physics_weights
        ),

        boundary_weight=np.array(
            episode_boundary_weights
        ),

        physics_loss=np.array(
            episode_physics_loss
        ),

        boundary_loss=np.array(
            episode_boundary_loss
        ),

        validation_score=np.array(
            episode_validation_score
        ),

        alpha_epsilon=np.array(
            episode_alpha_epsilon
        ),

        beta_epsilon=np.array(
            episode_beta_epsilon
        ),
    )

    # ========================================================
    # Best result selected ONLY by the physics-based
    # validation criterion. The reference solution is NOT
    # used for episode/model selection.
    # ========================================================

    best_index = int(
        np.argmin(
            episode_validation_score
        )
    )

    print()
    print("=" * 70)
    print(
        "BRATU TWO-AGENT DDQN FINAL RESULT"
    )
    print("=" * 70)

    print(
        f"Best Episode          : "
        f"{episode_numbers[best_index]}"
    )

    print(
        f"Best Validation Score : "
        f"{episode_validation_score[best_index]:.6e}"
    )

    print(
        f"Best MSE              : "
        f"{episode_mse[best_index]:.6e}"
    )

    print(
        f"Best Relative L2      : "
        f"{episode_relative_l2[best_index]:.6e}"
    )

    print(
        f"Best Physics Weight   : "
        f"{episode_physics_weights[best_index]:.4f}"
    )

    print(
        f"Best Boundary Weight  : "
        f"{episode_boundary_weights[best_index]:.4f}"
    )

    print()
    print(
        "Results saved to:"
    )
    print(output_file)

    print("=" * 70)


if __name__ == "__main__":
    main()