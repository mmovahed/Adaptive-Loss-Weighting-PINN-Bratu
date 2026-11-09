from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main():

    # =====================================================
    # Paths
    # =====================================================

    project_root = Path(__file__).resolve().parent.parent

    results_dir = project_root / "results"

    data_file = results_dir / "bratu_results.npz"

    # =====================================================
    # Load results
    # =====================================================

    data = np.load(data_file)

    x = data["x"]
    y_pred = data["y_pred"]
    y_ref = data["y_ref"]

    total_loss = data["history_total"]
    physics_loss = data["history_physics"]
    boundary_loss = data["history_boundary"]

    # =====================================================
    # 1. Solution
    # =====================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x,
        y_pred,
        label="PINN",
    )

    ax.plot(
        x,
        y_ref,
        "--",
        label="Reference",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Bratu Problem - Solution")

    ax.legend()
    ax.grid(True)

    fig.savefig(
        results_dir / "bratu_solution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =====================================================
    # 2. Absolute Error
    # =====================================================

    absolute_error = np.abs(
        y_pred - y_ref
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x,
        absolute_error,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("|u_PINN - u_ref|")
    ax.set_title("Bratu Problem - Absolute Error")

    ax.grid(True)

    fig.savefig(
        results_dir / "bratu_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =====================================================
    # 3. Training Loss
    # =====================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        total_loss,
        label="Total Loss",
    )

    ax.semilogy(
        physics_loss,
        label="Physics Loss",
    )

    ax.semilogy(
        boundary_loss,
        label="Boundary Loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN Training History")

    ax.legend()
    ax.grid(True)

    fig.savefig(
        results_dir / "bratu_training.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Plots successfully saved:")
    print(results_dir / "bratu_solution.png")
    print(results_dir / "bratu_error.png")
    print(results_dir / "bratu_training.png")


if __name__ == "__main__":
    main()