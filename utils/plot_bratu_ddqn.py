import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

results_dir = PROJECT_ROOT / "results"
figures_dir = PROJECT_ROOT / "figures"

figures_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load DDQN results
# ============================================================

data = np.load(results_dir / "bratu_ddqn_results.npz")

episode = data["episode"]
reward = data["reward"]
mse = data["mse"]
physics_weight = data["physics_weight"]
boundary_weight = data["boundary_weight"]
epsilon = data["epsilon"]

# ============================================================
# Figure style
# ============================================================

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11

# ============================================================
# 1. Reward per Episode
# ============================================================

plt.figure()

plt.plot(
    episode,
    reward,
    linewidth=2,
    marker="o",
)

plt.title("DDQN Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.tight_layout()
plt.savefig(figures_dir / "ddqn_reward.png", dpi=300)
plt.close()

# ============================================================
# 2. MSE per Episode (Log Scale)
# ============================================================

plt.figure()

plt.semilogy(
    episode,
    mse,
    linewidth=2,
    marker="o",
)

plt.title("PINN Validation MSE during DDQN Training")
plt.xlabel("Episode")
plt.ylabel("MSE (log scale)")
plt.grid(True, which="both")

plt.tight_layout()
plt.savefig(figures_dir / "ddqn_mse.png", dpi=300)
plt.close()

# ============================================================
# 3. Learned Loss Weights
# ============================================================

plt.figure()

plt.plot(
    episode,
    physics_weight,
    linewidth=2,
    marker="o",
    label="Physics Weight",
)

plt.plot(
    episode,
    boundary_weight,
    linewidth=2,
    marker="s",
    label="Boundary Weight",
)

plt.title("Adaptive Loss Weights Learned by DDQN")
plt.xlabel("Episode")
plt.ylabel("Weight Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(figures_dir / "ddqn_weights.png", dpi=300)
plt.close()

# ============================================================
# 4. Epsilon Decay
# ============================================================

plt.figure()

plt.plot(
    episode,
    epsilon,
    linewidth=2,
    marker="o",
)

plt.title("DDQN Exploration Rate (Epsilon)")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)

plt.tight_layout()
plt.savefig(figures_dir / "ddqn_epsilon.png", dpi=300)
plt.close()

# ============================================================
# Summary
# ============================================================

best_episode = np.argmin(mse)

print("=" * 60)
print("DDQN RESULTS SUMMARY")
print("=" * 60)

print(f"Best Episode           : {episode[best_episode]}")
print(f"Best MSE               : {mse[best_episode]:.6e}")
print(f"Physics Weight (best)  : {physics_weight[best_episode]:.2f}")
print(f"Boundary Weight (best) : {boundary_weight[best_episode]:.2f}")

print("\nFinal Episode Results")
print(f"Final MSE              : {mse[-1]:.6e}")
print(f"Final Physics Weight   : {physics_weight[-1]:.2f}")
print(f"Final Boundary Weight  : {boundary_weight[-1]:.2f}")
print(f"Final Epsilon          : {epsilon[-1]:.4f}")

print("\nFigures saved to:")
print(figures_dir)

print("=" * 60)