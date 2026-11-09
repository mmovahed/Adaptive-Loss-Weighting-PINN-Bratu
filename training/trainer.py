import torch


class PINNTrainer:
    def __init__(
        self,
        model,
        problem,
        learning_rate=1e-3,
        physics_weight=1.0,
        boundary_weight=1.0,
        device="cpu",
    ):
        self.model = model.to(device)
        self.problem = problem

        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.history = {
            "total_loss": [],
            "physics_loss": [],
            "boundary_loss": [],
        }
        
        self.x_collocation = None

    def compute_loss(self, n_collocation):
        # Create collocation points only once per training run
        if self.x_collocation is None:
            self.x_collocation = (
                self.problem.sample_collocation_points(
                    n_collocation,
                    device=self.device,
                )
            )

        # Physics loss
        residual = self.problem.residual(
            self.model,
            self.x_collocation,
        )

        physics_loss = torch.mean(
            residual ** 2
        )

        # Boundary loss
        boundary_loss = (
            self.problem.boundary_loss(
                self.model
            )
        )

        # Total loss
        total_loss = (
            self.physics_weight * physics_loss
            + self.boundary_weight * boundary_loss
        )

        return (
            total_loss,
            physics_loss,
            boundary_loss,
        )

    # ========================================================
    # Adam
    # ========================================================

    def train_adam(
        self,
        epochs=5000,
        n_collocation=100,
        print_every=500,
    ):
        self.model.train()

        for epoch in range(1, epochs + 1):

            self.optimizer.zero_grad()

            (
                total_loss,
                physics_loss,
                boundary_loss,
            ) = self.compute_loss(
                n_collocation
            )

            total_loss.backward()

            self.optimizer.step()

            self.history["total_loss"].append(
                total_loss.item()
            )

            self.history["physics_loss"].append(
                physics_loss.item()
            )

            self.history["boundary_loss"].append(
                boundary_loss.item()
            )

            if epoch % print_every == 0:
                print(
                    f"Adam Epoch {epoch:5d} | "
                    f"Total: {total_loss.item():.6e} | "
                    f"Physics: {physics_loss.item():.6e} | "
                    f"Boundary: {boundary_loss.item():.6e}"
                )

        return self.history

    # ========================================================
    # L-BFGS
    # ========================================================

    def train_lbfgs(
        self,
        n_collocation=100,
        max_iter=1000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    ):
        self.model.train()

        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        iteration = [0]

        def closure():

            optimizer.zero_grad()

            (
                total_loss,
                physics_loss,
                boundary_loss,
            ) = self.compute_loss(
                n_collocation
            )

            total_loss.backward()

            iteration[0] += 1

            if (
                iteration[0] == 1
                or iteration[0] % 100 == 0
            ):
                print(
                    f"L-BFGS Iteration "
                    f"{iteration[0]:4d} | "
                    f"Total: {total_loss.item():.6e} | "
                    f"Physics: {physics_loss.item():.6e} | "
                    f"Boundary: {boundary_loss.item():.6e}"
                )

            return total_loss

        optimizer.step(closure)

        return self.history