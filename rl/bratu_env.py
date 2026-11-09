import numpy as np


class BratuWeightEnv:

    def __init__(
        self,
        initial_physics_weight=1.0,
        initial_boundary_weight=1.0,
        min_weight=0.1,
        max_weight=3.0,
        step_size=0.1,
    ):
        self.initial_physics_weight = initial_physics_weight
        self.initial_boundary_weight = initial_boundary_weight

        self.min_weight = min_weight
        self.max_weight = max_weight
        self.step_size = step_size

        # 5 actions:
        # 0 = no change
        # 1 = decrease physics
        # 2 = increase physics
        # 3 = decrease boundary
        # 4 = increase boundary
        self.n_actions = 5

        self.reset()

    def reset(self):
        self.physics_weight = self.initial_physics_weight
        self.boundary_weight = self.initial_boundary_weight

        self.previous_loss = None

        return self._get_state()

    def _get_state(self):
        """
        State:
        [physics_weight,
         boundary_weight,
         previous_loss]
        """

        loss = (
            0.0
            if self.previous_loss is None
            else self.previous_loss
        )

        return np.array(
            [
                self.physics_weight,
                self.boundary_weight,
                loss,
            ],
            dtype=np.float32,
        )

    def step(self, action, current_loss):
        """
        Apply an action and calculate reward.

        Reward is based on improvement in loss.
        """

        old_loss = self.previous_loss

        # ----------------------------------------------------
        # Apply action
        # ----------------------------------------------------

        if action == 1:
            self.physics_weight -= self.step_size

        elif action == 2:
            self.physics_weight += self.step_size

        elif action == 3:
            self.boundary_weight -= self.step_size

        elif action == 4:
            self.boundary_weight += self.step_size

        # ----------------------------------------------------
        # Keep weights inside valid range
        # ----------------------------------------------------

        self.physics_weight = np.clip(
            self.physics_weight,
            self.min_weight,
            self.max_weight,
        )

        self.boundary_weight = np.clip(
            self.boundary_weight,
            self.min_weight,
            self.max_weight,
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        if old_loss is None:
            reward = 0.0
        else:
            reward = old_loss - current_loss

        self.previous_loss = current_loss

        state = self._get_state()

        done = False

        return state, reward, done, {
            "physics_weight": self.physics_weight,
            "boundary_weight": self.boundary_weight,
            "loss": current_loss,
        }