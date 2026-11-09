import numpy as np
import torch
from scipy.integrate import solve_bvp


class BratuProblem:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.x_min = 0.0
        self.x_max = 1.0

    def residual(self, model, x):
        x = x.clone().detach().requires_grad_(True)
        u = model(x)

        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx,
            inputs=x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
        )[0]

        return d2u_dx2 + self.lam * torch.exp(u)

    def boundary_values(self, model):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x_boundary = torch.tensor(
            [[self.x_min], [self.x_max]],
            dtype=dtype,
            device=device,
        )
        return model(x_boundary)

    def boundary_loss(self, model):
        """Mean squared homogeneous-Dirichlet error over both endpoints.

        This matches 1/2 * (u(0)^2 + u(1)^2) for the two Bratu endpoints.
        """
        values = self.boundary_values(model)
        return torch.mean(values ** 2)

    def sample_collocation_points(self, n_points, device="cpu"):
        """Return exactly n_points fixed interior collocation points."""
        if int(n_points) < 1:
            raise ValueError("n_points must be positive")
        return torch.linspace(
            self.x_min,
            self.x_max,
            int(n_points) + 2,
            device=device,
        ).reshape(-1, 1)[1:-1]

    def reference_solution(self, n_points=1000):
        x = np.linspace(self.x_min, self.x_max, 100)
        y_guess = np.zeros((2, x.size))

        def ode(x, y):
            u = y[0]
            du_dx = y[1]
            d2u_dx2 = -self.lam * np.exp(u)
            return np.vstack([du_dx, d2u_dx2])

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        solution = solve_bvp(ode, bc, x, y_guess)
        if not solution.success:
            raise RuntimeError("Reference solver failed: {}".format(solution.message))

        x_ref = np.linspace(self.x_min, self.x_max, n_points)
        y_ref = solution.sol(x_ref)[0]
        return x_ref, y_ref
