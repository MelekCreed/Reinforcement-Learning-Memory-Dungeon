import numpy as np
from dungeon.generator import solution

class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.choice(np.flatnonzero(obs.action_mask())))

class OracleAgent:
    """Explicitly privileged upper bound. Never used by training or the demo policy."""
    def __init__(self, dungeon):
        self.plan = iter(solution(dungeon))

    def act(self, obs):
        return next(self.plan)
