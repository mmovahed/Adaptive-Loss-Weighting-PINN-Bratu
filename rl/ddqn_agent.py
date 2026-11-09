import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Q-Network
# ============================================================

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# DDQN Agent
# ============================================================

class DDQNAgent:

    def __init__(
        self,
        state_dim=3,
        action_dim=5,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=10000,
        target_update_frequency=100,
        device="cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.device = device

        # ----------------------------------------------------
        # Online network
        # ----------------------------------------------------

        self.online_network = QNetwork(
            state_dim,
            action_dim,
        ).to(device)

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------

        self.target_network = QNetwork(
            state_dim,
            action_dim,
        ).to(device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        # ----------------------------------------------------
        # Replay memory
        # ----------------------------------------------------

        self.memory = deque(
            maxlen=memory_size
        )

        # ----------------------------------------------------
        # Target network update
        # ----------------------------------------------------

        self.target_update_frequency = (
            target_update_frequency
        )

        self.train_steps = 0

    # ========================================================
    # Action selection
    # ========================================================

    def select_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(
                self.action_dim
            )

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.online_network(
                state_tensor
            )

        return int(
            torch.argmax(q_values, dim=1).item()
        )

    # ========================================================
    # Store transition
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    # ========================================================
    # Training
    # ========================================================

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory,
            self.batch_size,
        )

        states = torch.tensor(
            np.array(
                [x[0] for x in batch]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            [x[1] for x in batch],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            [x[2] for x in batch],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            np.array(
                [x[3] for x in batch]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            [x[4] for x in batch],
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Current Q values
        # ----------------------------------------------------

        current_q = self.online_network(
            states
        ).gather(
            1,
            actions,
        ).squeeze(1)

        # ----------------------------------------------------
        # DDQN target
        # ----------------------------------------------------

        with torch.no_grad():

            # Online network chooses action
            next_actions = torch.argmax(
                self.online_network(next_states),
                dim=1,
                keepdim=True,
            )

            # Target network evaluates action
            next_q = self.target_network(
                next_states
            ).gather(
                1,
                next_actions,
            ).squeeze(1)

            target_q = rewards + (
                self.gamma
                * next_q
                * (1.0 - dones)
            )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = nn.functional.mse_loss(
            current_q,
            target_q,
        )

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.train_steps += 1

        # ----------------------------------------------------
        # Target update
        # ----------------------------------------------------

        if (
            self.train_steps
            % self.target_update_frequency
            == 0
        ):

            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        # ----------------------------------------------------
        # Epsilon decay
        # ----------------------------------------------------

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

        return float(loss.item())