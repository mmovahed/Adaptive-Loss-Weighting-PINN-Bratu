import torch


def mse(y_pred, y_true):
    return torch.mean(
        (y_pred - y_true) ** 2
    ).item()


def mae(y_pred, y_true):
    return torch.mean(
        torch.abs(y_pred - y_true)
    ).item()


def max_absolute_error(y_pred, y_true):
    return torch.max(
        torch.abs(y_pred - y_true)
    ).item()


def relative_l2_error(y_pred, y_true):
    numerator = torch.norm(
        y_pred - y_true
    )

    denominator = torch.norm(
        y_true
    )

    return (
        numerator / denominator
    ).item()


def boundary_error(model):
    device = next(model.parameters()).device

    x_left = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
        device=device,
    )

    x_right = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        u_left = model(x_left)
        u_right = model(x_right)

    return {
        "left": abs(u_left.item()),
        "right": abs(u_right.item()),
    }