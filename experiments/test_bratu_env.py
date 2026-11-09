import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.bratu_env import BratuWeightEnv


env = BratuWeightEnv()

state = env.reset()

print("Initial state:")
print(state)

state, reward, done, info = env.step(
    action=2,
    current_loss=0.5,
)

print("\nAfter action:")
print(state)

print("\nReward:")
print(reward)

print("\nInfo:")
print(info)