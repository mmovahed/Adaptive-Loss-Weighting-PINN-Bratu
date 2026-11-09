import sys
from pathlib import Path
import random

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pinn import PINN
from problems.bratu import BratuProblem
from training.trainer import PINNTrainer
from utils.metrics import (
    mse,
    mae,
    max_absolute_error,
    relative_l2_error,
)

from configs.bratu_config import (
    LAMBDA,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    N_COLLOCATION,
    N_TEST,
    TOTAL_PINN_STEPS,
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
# Run one fixed-weight experiment
# ============================================================

def run_experiment(
    physics_weight,
    boundary_weight,
    device,
):
    # --------------------------------------------------------
    # Problem
    # --------------------------------------------------------

    problem = BratuProblem(
        lam=LAMBDA
    )

    # --------------------------------------------------------
    # PINN
    # --------------------------------------------------------

    model = PINN(
        input_dim=1,
        output_dim=1,
        hidden_layers=HIDDEN_LAYERS,
    ).to(device)

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = PINNTrainer(
        model=model,
        problem=problem,
        learning_rate=LEARNING_RATE,
        physics_weight=physics_weight,
        boundary_weight=boundary_weight,
        device=device,
    )

    # --------------------------------------------------------
    # Adam
    # --------------------------------------------------------

    trainer.train_adam(
        epochs=TOTAL_PINN_STEPS,
        n_collocation=N_COLLOCATION,
        print_every=1000,
    )

    # --------------------------------------------------------
    # Test grid
    # --------------------------------------------------------

    x_test = torch.linspace(
        0.0,
        1.0,
        N_TEST,
        device=device,
    ).reshape(-1, 1)

    model.eval()

    with torch.no_grad():
        y_pred = model(x_test)

    # --------------------------------------------------------
    # Reference solution
    # --------------------------------------------------------

    _, y_ref = problem.reference_solution(
        n_points=N_TEST
    )

    y_ref_tensor = torch.tensor(
        y_ref,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return {
        "mse": mse(
            y_pred,
            y_ref_tensor,
        ),
        "mae": mae(
            y_pred,
            y_ref_tensor,
        ),
        "max_error": max_absolute_error(
            y_pred,
            y_ref_tensor,
        ),
        "relative_l2": relative_l2_error(
            y_pred,
            y_ref_tensor,
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Weight configurations
    # --------------------------------------------------------

    weight_configs = [
        (1.0, 1.0),
        (0.3, 1.0),
        (3.0, 1.0),
        (1.0, 0.3),
        (1.0, 3.0),
    ]

    results = []

    print("\n")
    print("=" * 75)
    print("BRATU FIXED-WEIGHT SEARCH")
    print("=" * 75)

    print(
        f"Lambda       : {LAMBDA}"
    )

    print(
        f"Architecture : "
        f"1-{HIDDEN_LAYERS[0]}-"
        f"{HIDDEN_LAYERS[1]}-"
        f"{HIDDEN_LAYERS[2]}-1"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Collocation  : {N_COLLOCATION}"
    )

    print(
        f"PINN steps   : {TOTAL_PINN_STEPS}"
    )

    print("=" * 75)

    for physics_weight, boundary_weight in weight_configs:

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"Testing alpha={physics_weight}, "
            f"beta={boundary_weight}"
        )

        print(
            "-" * 75
        )

        # ----------------------------------------------------
        # Same initialization for each configuration
        # ----------------------------------------------------

        set_seed(42)

        metrics = run_experiment(
            physics_weight=physics_weight,
            boundary_weight=boundary_weight,
            device=device,
        )

        result = {
            "alpha": physics_weight,
            "beta": boundary_weight,
            **metrics,
        }

        results.append(result)

        print("\nResult:")

        print(
            f"MSE                : "
            f"{metrics['mse']:.6e}"
        )

        print(
            f"MAE                : "
            f"{metrics['mae']:.6e}"
        )

        print(
            f"Max Absolute Error : "
            f"{metrics['max_error']:.6e}"
        )

        print(
            f"Relative L2 Error  : "
            f"{metrics['relative_l2']:.6e}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("FIXED-WEIGHT SUMMARY")
    print("=" * 75)

    print(
        f"{'Alpha':>8} "
        f"{'Beta':>8} "
        f"{'MSE':>15} "
        f"{'MAE':>15} "
        f"{'Max Error':>15} "
        f"{'Relative L2':>15}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['alpha']:8.1f} "
            f"{result['beta']:8.1f} "
            f"{result['mse']:15.6e} "
            f"{result['mae']:15.6e} "
            f"{result['max_error']:15.6e} "
            f"{result['relative_l2']:15.6e}"
        )

    # --------------------------------------------------------
    # Select best configuration
    # --------------------------------------------------------

    best = min(
        results,
        key=lambda x: x["relative_l2"]
    )

    print("\n")
    print("=" * 75)
    print("BEST FIXED-TUNED CONFIGURATION")
    print("=" * 75)

    print(
        f"Alpha (Physics) : "
        f"{best['alpha']}"
    )

    print(
        f"Beta (Boundary) : "
        f"{best['beta']}"
    )

    print(
        f"MSE             : "
        f"{best['mse']:.6e}"
    )

    print(
        f"Relative L2     : "
        f"{best['relative_l2']:.6e}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()