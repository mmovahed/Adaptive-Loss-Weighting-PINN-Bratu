"""Small deterministic checks for the adaptive-weighting implementations.

Run with:
    python experiments/test_adaptive_weighting.py
No Bratu training is required.
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptive_weighting import (
    LearningRateAnnealingBalancer,
    ReLoBRaLoBalancer,
)


def assert_close(actual, expected, tol=1e-10, label="value"):
    if not np.allclose(actual, expected, rtol=tol, atol=tol):
        raise AssertionError("{} mismatch: {} != {}".format(label, actual, expected))


def test_lra_reference_update():
    # One weight tensor, no bias. At w=2:
    # Lf=(2w)^2 -> dLf/dw=8w=16
    # Lb=w^2. With lambda=1, d(lambda*Lb)/dw=2w=4.
    # lambda_hat=4 and smoothing=.9 gives new lambda=1.3.
    w = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
    balancer = LearningRateAnnealingBalancer(
        smoothing=0.9, initial_boundary_weight=1.0
    )
    physics_loss = (2.0 * w).pow(2).sum()
    boundary_loss = w.pow(2).sum()
    updated = balancer.update(physics_loss, boundary_loss, [w])
    assert_close(updated, 1.3, label="LRA first update")
    assert_close(balancer.last_raw_weight, 4.0, label="LRA lambda_hat")

    # Verify that the CURRENT weight is part of the denominator on update 2.
    physics_loss = (2.0 * w).pow(2).sum()
    boundary_loss = w.pow(2).sum()
    updated2 = balancer.update(physics_loss, boundary_loss, [w])
    raw2 = 16.0 / (1.3 * 4.0)
    expected2 = 0.9 * 1.3 + 0.1 * raw2
    assert_close(balancer.last_raw_weight, raw2, label="LRA weighted denominator")
    assert_close(updated2, expected2, label="LRA second update")


def _balanced(current, reference, temperature):
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    logits = current / (temperature * reference + 1e-12)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return len(current) * e / np.sum(e)


def test_relobralo_reference_bootstrap():
    balancer = ReLoBRaLoBalancer(
        n_losses=2,
        alpha=0.9,
        rho_probability=1.0,  # deterministic: always keep historical branch
        temperature=0.5,
        random_seed=123,
    )

    l0 = np.asarray([2.0, 1.0])
    w0 = balancer.update(l0)
    # Reference code's first-call alpha=1 and rho=1 keeps lambda=(1,1).
    assert_close(w0, np.ones(2), label="ReLoBRaLo bootstrap step 0")
    assert_close(balancer.last_effective_alpha, 1.0, label="bootstrap alpha0")

    l1 = np.asarray([1.0, 2.0])
    w1 = balancer.update(l1)
    expected1 = _balanced(l1, l0, 0.5)
    # Second-call alpha=0: use relative-to-previous softmax directly.
    assert_close(w1, expected1, tol=1e-9, label="ReLoBRaLo bootstrap step 1")
    assert_close(balancer.last_effective_alpha, 0.0, label="bootstrap alpha1")

    l2 = np.asarray([0.8, 1.5])
    step_weights = _balanced(l2, l1, 0.5)
    expected2 = 0.9 * expected1 + 0.1 * step_weights
    w2 = balancer.update(l2)
    assert_close(w2, expected2, tol=1e-9, label="ReLoBRaLo regular update")
    assert_close(np.sum(w2), 2.0, tol=1e-10, label="ReLoBRaLo weight sum")


def main():
    test_lra_reference_update()
    test_relobralo_reference_bootstrap()
    print("PASS: LRA reference update")
    print("PASS: ReLoBRaLo reference bootstrap/update")
    print("2/2 adaptive-weighting checks passed")


if __name__ == "__main__":
    main()
