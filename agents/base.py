"""One interface for all policies; hidden state is owned by the episode runner."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.distributions import Categorical
from .encoders import INPUT_DIM, Encoder
from dungeon.observations import FEATURE_DIM

@dataclass(frozen=True)
class PolicyConfig:
    agent: str = "gru"
    hidden: int = 64
    history: int = 4

class Policy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        if config.agent not in ("no_memory", "history", "gru", "lstm"):
            raise ValueError("Unknown agent")
        if config.history < 1 or config.hidden < 1:
            raise ValueError("Invalid network dimensions")
        self.config = config
        n = FEATURE_DIM if config.agent == "no_memory" else INPUT_DIM
        self.encoder = Encoder(n * (config.history if config.agent == "history" else 1), config.hidden)
        if config.agent in ("gru", "lstm"):
            self.core = (nn.GRU if config.agent == "gru" else nn.LSTM)(config.hidden, config.hidden)
        self.actor = nn.Linear(config.hidden, 13)
        self.critic = nn.Linear(config.hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def initial_state(self, batch: int, device=None):
        device = device or next(self.parameters()).device
        kind = self.config.agent
        if kind == "no_memory":
            return None
        if kind == "history":
            return torch.zeros(self.config.history-1, batch, INPUT_DIM, device=device)
        h = torch.zeros(1, batch, self.config.hidden, device=device)
        return (h, h.clone()) if kind == "lstm" else h

    def forward(self, x, state=None):
        """x: [time, batch, input]. Recompute full trajectories for PPO BPTT."""
        if state is None:
            state = self.initial_state(x.shape[1], x.device)
        kind = self.config.agent
        if kind == "no_memory":
            y = self.encoder(x[..., :FEATURE_DIM])
        elif kind == "history":
            k = self.config.history
            sequence = torch.cat((state, x), dim=0)
            windows = torch.cat([sequence[i:i+x.shape[0]] for i in range(k)], dim=-1)
            y = self.encoder(windows)
            state = sequence[-(k-1):] if k > 1 else sequence[:0]
        else:
            y, state = self.core(self.encoder(x), state)
        return self.actor(y), self.critic(y).squeeze(-1), state

    @staticmethod
    def distribution(logits, mask):
        return Categorical(logits=logits.masked_fill(~mask, -1e9))

    def save(self, path, metadata):
        torch.save({"policy_config": asdict(self.config), "state_dict": self.state_dict(), "metadata": metadata}, path)

def load_policy(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    policy = Policy(PolicyConfig(**data["policy_config"])).to(device)
    policy.load_state_dict(data["state_dict"])
    policy.eval()
    return policy, data["metadata"]

def clone_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(x.detach().clone() for x in state)
    return state.detach().clone()
