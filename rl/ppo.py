from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 0.0007
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

def update(policy, optimizer, rollout, cfg):
    adv, returns = rollout.advantages(cfg.gamma, cfg.gae_lambda)
    valid = rollout.active
    selected = adv[valid]
    adv = (adv - selected.mean()) / (selected.std(unbiased=False)+1e-8)
    stats = {}
    # No shuffled timesteps: each epoch replays the entire episode from zero
    # state under CURRENT parameters, preserving causal recurrent gradients.
    for _ in range(cfg.epochs):
        logits, values, _ = policy(rollout.inputs)
        dist = policy.distribution(logits, rollout.masks)
        logp = dist.log_prob(rollout.actions)
        ratio = (logp-rollout.log_probs).exp()
        policy_loss = -torch.minimum(ratio*adv, ratio.clamp(1-cfg.clip, 1+cfg.clip)*adv)[valid].mean()
        value_loss = (values-returns).square()[valid].mean()
        entropy = dist.entropy()[valid].mean()
        loss = policy_loss + cfg.value_coef*value_loss - cfg.entropy*entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        stats = {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                 "entropy": entropy.item(), "gradient_norm": float(grad),
                 "approx_kl": ((ratio-1)-(logp-rollout.log_probs))[valid].mean().item()}
    return stats
