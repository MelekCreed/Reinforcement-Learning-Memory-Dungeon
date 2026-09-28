from dataclasses import dataclass
import torch

@dataclass
class Rollout:
    inputs: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    active: torch.Tensor
    terminated: torch.Tensor
    bootstrap: torch.Tensor
    metrics: list[dict]

    def advantages(self, gamma=0.99, lam=0.95):
        """Terminal zero bootstrap; time limits bootstrap V(final observation)."""
        adv = torch.zeros_like(self.rewards)
        next_value = self.bootstrap
        carry = torch.zeros_like(next_value)
        for t in reversed(range(len(adv))):
            continuation = (~self.terminated[t]).float()
            delta = self.rewards[t] + gamma * next_value * continuation - self.values[t]
            carry = (delta + gamma * lam * continuation * carry) * self.active[t]
            adv[t] = carry
            # Preserve bootstrap through padding for episodes shorter than T.
            next_value = torch.where(self.active[t], self.values[t], next_value)
        return adv, adv + self.values
