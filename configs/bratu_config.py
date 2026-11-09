# ============================================================
# Shared Bratu adaptive-weighting benchmark configuration.
# Python 3.7 compatible. Environment variables can override
# defaults for smoke tests; Cluster PBS scripts pin final values.
# ============================================================

import os


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_str(name, default):
    return os.environ.get(name, default)


PROTOCOL_VERSION = _env_str("BRATU_PROTOCOL_VERSION", "bratu_adaptive_benchmark_v2")

LAMBDA = _env_float("BRATU_LAMBDA", 1.0)

# PINN architecture
HIDDEN_LAYERS = (64, 64, 64)

# Training
LEARNING_RATE = _env_float("BRATU_LEARNING_RATE", 1e-3)
N_COLLOCATION = _env_int("BRATU_N_COLLOCATION", 256)

# Validation / test. Validation points are INTERIOR points; BC is evaluated
# separately. Test points include both endpoints.
N_VALIDATION = _env_int("BRATU_N_VALIDATION", 512)
N_TEST = _env_int("BRATU_N_TEST", 2000)

# Total PINN parameter updates per candidate PINN.
TOTAL_PINN_STEPS = _env_int("BRATU_TOTAL_PINN_STEPS", 15000)

# DDQN
WARMUP_STEPS = _env_int("BRATU_WARMUP_STEPS", 200)
DECISION_INTERVAL = _env_int("BRATU_DECISION_INTERVAL", 20)
DDQN_TRAIN_EPISODES = _env_int("BRATU_DDQN_TRAIN_EPISODES", 5)

# DDQN loss-weight bounds
INITIAL_ALPHA = 1.0
INITIAL_BETA = 1.0
MIN_WEIGHT = 0.01
MAX_WEIGHT = 100.0

# Learning-rate annealing (Wang, Teng & Perdikaris), matched to the
# authors' released GradientPathologiesPINNs reference implementation.
# There the boundary-gradient statistic is computed from the CURRENTLY
# WEIGHTED boundary loss, the previous adaptive coefficient is retained with
# factor 0.9, and the coefficient is refreshed every 10 optimizer steps.
LRA_SMOOTHING = _env_float(
    "BRATU_LRA_SMOOTHING",
    _env_float("BRATU_LRA_DECAY", 0.9),  # backward-compatible env alias
)
LRA_DECAY = LRA_SMOOTHING  # backward-compatible Python alias
LRA_UPDATE_INTERVAL = _env_int("BRATU_LRA_UPDATE_INTERVAL", 10)

# ReLoBRaLo (Bischof & Kraus): exact stochastic random-lookback update.
# These are literature-derived, not tuned on the Bratu test set. The defaults
# correspond to the robust regime used in the authors' published benchmarks.
RELOBRALO_ALPHA = _env_float("BRATU_RELOBRALO_ALPHA", 0.999)
RELOBRALO_RHO = _env_float(
    "BRATU_RELOBRALO_RHO",
    _env_float("BRATU_RELOBRALO_BETA", 0.9999),  # legacy env compatibility
)
RELOBRALO_TEMPERATURE = _env_float(
    "BRATU_RELOBRALO_TEMPERATURE",
    _env_float("BRATU_RELOBRALO_TAU", 0.1),  # legacy env compatibility
)

# Logging
WEIGHT_LOG_INTERVAL = _env_int("BRATU_WEIGHT_LOG_INTERVAL", 20)

# Matched computational-budget / model-selection protocol.
# Every compared method receives the same candidate PINN initializations and
# the same validation-only selector. The independent reference solution is
# never used to select candidates.
SELECTION_CANDIDATES = _env_int("BRATU_SELECTION_CANDIDATES", 5)
CANDIDATE_SEED_STRIDE = _env_int("BRATU_CANDIDATE_SEED_STRIDE", 1000003)
SELECTION_PROTOCOL = "matched_best_of_n_validation_v2"

# Static comparison. This locked unequal-weight pair is retained as a
# contextual static comparator. It is NOT tuned or re-tuned inside benchmark V2
# and is not used to choose any adaptive-method hyperparameters.
FIXED_SELECTED_ALPHA = _env_float("BRATU_FIXED_SELECTED_ALPHA", 1.0)
FIXED_SELECTED_BETA = _env_float("BRATU_FIXED_SELECTED_BETA", 3.0)

# Final outer seeds
SEEDS = [42, 123, 456, 789, 1000]
