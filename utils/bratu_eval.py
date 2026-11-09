import json
import random
from pathlib import Path

import numpy as np
import torch

from utils.metrics import mse, mae, max_absolute_error, relative_l2_error


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_validation_grid(n_validation, device):
    """Exactly n_validation fixed INTERIOR points."""
    return torch.linspace(
        0.0,
        1.0,
        int(n_validation) + 2,
        device=device,
    ).reshape(-1, 1)[1:-1]


def validation_losses(model, problem, x_val):
    """Unweighted validation components; no reference solution is used."""
    model.eval()
    physics_residual = problem.residual(model, x_val)
    physics_loss = torch.mean(physics_residual ** 2)

    with torch.no_grad():
        boundary_loss = problem.boundary_loss(model)

    return float(physics_loss.item()), float(boundary_loss.item())


def _reference_tensors(problem, n_test, device):
    x_test = torch.linspace(0.0, 1.0, n_test, device=device).reshape(-1, 1)
    x_ref, y_ref = problem.reference_solution(n_points=n_test)
    y_ref_tensor = torch.tensor(
        y_ref,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    return x_test, x_ref, y_ref, y_ref_tensor


def _evaluate_with_reference(model, x_test, x_ref, y_ref, y_ref_tensor):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)

    return {
        "mse": mse(y_pred, y_ref_tensor),
        "mae": mae(y_pred, y_ref_tensor),
        "max_error": max_absolute_error(y_pred, y_ref_tensor),
        "relative_l2": relative_l2_error(y_pred, y_ref_tensor),
        "x": x_ref,
        "y_pred": y_pred.detach().cpu().numpy().flatten(),
        "y_ref": y_ref,
    }


def evaluate_model(model, problem, n_test, device):
    x_test, x_ref, y_ref, y_ref_tensor = _reference_tensors(
        problem, n_test, device
    )
    return _evaluate_with_reference(
        model, x_test, x_ref, y_ref, y_ref_tensor
    )


def evaluate_candidate_models(models, problem, n_test, device):
    """Evaluate all candidate PINNs only AFTER validation-only selection.

    The common reference solution is solved once. These metrics are for
    post-selection reporting/robustness analysis and never affect selection.
    """
    x_test, x_ref, y_ref, y_ref_tensor = _reference_tensors(
        problem, n_test, device
    )
    return [
        _evaluate_with_reference(model, x_test, x_ref, y_ref, y_ref_tensor)
        for model in models
    ]


def run_result_path(project_root, method, lam, seed):
    out_dir = Path(project_root) / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    lam_text = str(lam).replace(".", "p")
    return out_dir / "{}_lambda_{}_seed_{}.npz".format(method, lam_text, seed)


def print_result_json(result):
    serializable = {}
    for key, value in result.items():
        if isinstance(value, (int, float, str)) or value is None:
            serializable[key] = value
    print("RESULT_JSON: " + json.dumps(serializable, sort_keys=True))
