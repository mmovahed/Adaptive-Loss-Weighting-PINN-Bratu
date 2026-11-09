import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rl.ddqn_agent import DDQNAgent


agent = DDQNAgent(
    state_dim=3,
    action_dim=5,
    device="cpu",
)

state = np.array(
    [1.0, 1.0, 0.0],
    dtype=np.float32,
)

print("Initial epsilon:", agent.epsilon)

action = agent.select_action(state)

print("Selected action:", action)

next_state = np.array(
    [1.1, 1.0, 0.5],
    dtype=np.float32,
)

agent.remember(
    state=state,
    action=action,
    reward=0.1,
    next_state=next_state,
    done=False,
)

print("Replay memory size:", len(agent.memory))

print("DDQN agent test completed successfully.")