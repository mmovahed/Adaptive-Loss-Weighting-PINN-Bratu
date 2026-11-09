import numpy as np
import torch


class LearningRateAnnealingBalancer:
    """Reference-code-faithful LRA for the two-term Bratu PINN.

    This follows the implementation released by Wang, Teng and Perdikaris
    with ``GradientPathologiesPINNs`` for a loss

        L = L_f + lambda_b L_b.

    At an LRA update, their reference code computes gradient statistics with
    respect to network *weight tensors* (biases are not included):

        lambda_hat = max_j max(|d L_f / d W_j|)
                     / mean_j mean(|d (lambda_b L_b) / d W_j|)

        lambda_b <- smoothing * lambda_b
                    + (1 - smoothing) * lambda_hat.

    Their released Helmholtz implementation uses smoothing=0.9 and updates
    the adaptive coefficient every 10 optimizer iterations. The caller owns
    the update frequency, which lets the same class be unit-tested directly.

    Note the weighted denominator. Some later descriptions/implementations
    write the denominator using the unweighted auxiliary loss. We intentionally
    match the authors' released reference implementation used with the paper.
    """

    def __init__(self, smoothing=0.9, initial_boundary_weight=1.0, eps=1e-12):
        smoothing = float(smoothing)
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.boundary_weight = float(initial_boundary_weight)
        self.eps = float(eps)

        self.last_raw_weight = float(initial_boundary_weight)
        self.last_numerator = np.nan
        self.last_denominator = np.nan
        self.n_updates = 0

    @staticmethod
    def _reference_weight_parameters(parameters):
        """Use matrix/tensor weights, matching the reference TF code.

        ``model.parameters()`` in PyTorch also yields 1-D bias tensors, while
        the original implementation differentiates only ``self.weights``.
        For a standard fully connected PINN, selecting ndim >= 2 reproduces
        that choice.
        """
        params = [p for p in parameters if p.requires_grad and p.ndim >= 2]
        if not params:
            # Defensive fallback for an unusual model with no matrix weights.
            params = [p for p in parameters if p.requires_grad]
        return params

    @staticmethod
    def _layer_max_abs(grads):
        values = [torch.max(torch.abs(g)) for g in grads if g is not None]
        if not values:
            return None
        return torch.max(torch.stack(values))

    @staticmethod
    def _mean_of_layer_means_abs(grads):
        values = [torch.mean(torch.abs(g)) for g in grads if g is not None]
        if not values:
            return None
        return torch.mean(torch.stack(values))

    def update(self, physics_loss, boundary_loss, parameters):
        params = self._reference_weight_parameters(list(parameters))

        physics_grads = torch.autograd.grad(
            physics_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        # This weighting inside the gradient is deliberate: it mirrors the
        # authors' released code, where loss_bcs already contains lambda_b.
        weighted_boundary = self.boundary_weight * boundary_loss
        boundary_grads = torch.autograd.grad(
            weighted_boundary,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        numerator_tensor = self._layer_max_abs(physics_grads)
        denominator_tensor = self._mean_of_layer_means_abs(boundary_grads)
        if numerator_tensor is None or denominator_tensor is None:
            return self.boundary_weight

        numerator = float(numerator_tensor.detach().item())
        denominator = float(denominator_tensor.detach().item())
        candidate = numerator / max(denominator, self.eps)

        self.last_numerator = numerator
        self.last_denominator = denominator
        self.last_raw_weight = candidate

        if np.isfinite(candidate):
            self.boundary_weight = (
                self.smoothing * self.boundary_weight
                + (1.0 - self.smoothing) * candidate
            )
        self.n_updates += 1
        return self.boundary_weight


class ReLoBRaLoBalancer:
    """Reference-code-faithful ReLoBRaLo state machine.

    This implements the update released by Bischof & Kraus in the authors'
    ``relative_balancing`` repository. For m loss terms, with current losses
    L_i(t) and reference losses L_i(t'), define

        lambda_bal_i(t,t') = m * softmax_i(L_i(t)/(T L_i(t'))).

    The regular update is

        lambda_i(t) = rho * alpha * lambda_i(t-1)
                      + (1-rho) * alpha * lambda_bal_i(t,0)
                      + (1-alpha) * lambda_bal_i(t,t-1),

    with rho sampled from Bernoulli(rho_probability).

    The authors' reference code uses a two-step bootstrap schedule for alpha:
    alpha=1 on the first call, alpha=0 on the second, and the configured alpha
    thereafter; the initial-lookback losses are stored on the second call.
    We reproduce that behavior exactly rather than silently replacing it with
    a simplified initialization.
    """

    def __init__(
        self,
        n_losses,
        alpha=0.999,
        rho_probability=0.9999,
        temperature=0.1,
        eps=1e-12,
        random_seed=None,
    ):
        self.n_losses = int(n_losses)
        self.alpha = float(alpha)
        self.rho_probability = float(rho_probability)
        self.temperature = float(temperature)
        self.eps = float(eps)

        if self.n_losses < 2:
            raise ValueError("ReLoBRaLo requires at least two loss terms")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if not 0.0 <= self.rho_probability <= 1.0:
            raise ValueError("rho_probability must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        # Matches the reference implementation's initial args: all lambda,
        # previous-loss and initial-loss entries begin at 1.
        self.weights = np.ones(self.n_losses, dtype=np.float64)
        self.previous_losses = np.ones(self.n_losses, dtype=np.float64)
        self.initial_losses = np.ones(self.n_losses, dtype=np.float64)
        self.rng = np.random.RandomState(random_seed)
        self.step_index = 0

        # Diagnostics for trajectory/audit output.
        self.last_rho = 1.0
        self.last_effective_alpha = 1.0
        self.last_step_weights = self.weights.copy()
        self.last_initial_weights = self.weights.copy()

    def _balanced_weights(self, current, reference):
        logits = current / (self.temperature * reference + self.eps)
        logits = logits - np.max(logits)
        exp_values = np.exp(np.clip(logits, -700.0, 0.0))
        denom = float(np.sum(exp_values))
        if not np.isfinite(denom) or denom <= self.eps:
            return np.ones(self.n_losses, dtype=np.float64)
        return self.n_losses * exp_values / denom

    def update(self, losses):
        current = np.asarray(losses, dtype=np.float64)
        if current.shape != (self.n_losses,):
            raise ValueError(
                "Expected {} losses, got shape {}".format(
                    self.n_losses, current.shape
                )
            )
        current = np.maximum(current, self.eps)

        # Exact bootstrap schedule used in the authors' train.py.
        if self.step_index == 0:
            effective_alpha = 1.0
        elif self.step_index == 1:
            effective_alpha = 0.0
        else:
            effective_alpha = self.alpha

        step_weights = self._balanced_weights(current, self.previous_losses)
        initial_weights = self._balanced_weights(current, self.initial_losses)
        rho = float(self.rng.uniform() < self.rho_probability)

        self.weights = (
            rho * effective_alpha * self.weights
            + (1.0 - rho) * effective_alpha * initial_weights
            + (1.0 - effective_alpha) * step_weights
        )

        # Softmax terms sum to m, and convex mixing should preserve that sum.
        # Renormalization only removes floating-point drift.
        weight_sum = float(np.sum(self.weights))
        if np.isfinite(weight_sum) and weight_sum > self.eps:
            self.weights = self.n_losses * self.weights / weight_sum

        self.previous_losses = current.copy()
        if self.step_index == 1:
            # Reference code stores l0 after the second update call.
            self.initial_losses = current.copy()

        self.last_rho = rho
        self.last_effective_alpha = effective_alpha
        self.last_step_weights = step_weights.copy()
        self.last_initial_weights = initial_weights.copy()
        self.step_index += 1
        return self.weights.copy()
