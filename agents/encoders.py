import torch
from torch import nn
from dungeon.observations import FEATURE_DIM

INPUT_DIM = FEATURE_DIM + 13

def policy_input(observation, previous_action: int | None = None):
    """Local features and previous action. No privileged environment input."""
    x = torch.zeros(INPUT_DIM)
    x[:FEATURE_DIM] = torch.from_numpy(observation.features())
    if previous_action is not None:
        x[FEATURE_DIM + previous_action] = 1
    return x

class Encoder(nn.Sequential):
    def __init__(self, inputs: int, hidden: int):
        super().__init__(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
