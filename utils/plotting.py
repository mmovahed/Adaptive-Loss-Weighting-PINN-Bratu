import matplotlib

# Use a non-interactive backend.
# This avoids GUI/backend issues when running from Spyder.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_solution(
    x,
    y_pred,
    y_true=None,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x,
        y_pred,
        label="PINN",
    )

    if y_true is not None:
        ax.plot(
            x,
            y_true,
            "--",
            label="Reference",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Bratu Problem - Solution")

    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_error(
    x,
    error,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x,
        error,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("|u_PINN - u_ref|")
    ax.set_title("Bratu Problem - Absolute Error")

    ax.grid(True)

    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_training_history(
    history,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        history["total_loss"],
        label="Total Loss",
    )

    ax.semilogy(
        history["physics_loss"],
        label="Physics Loss",
    )

    ax.semilogy(
        history["boundary_loss"],
        label="Boundary Loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN Training History")

    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)