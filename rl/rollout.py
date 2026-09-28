import torch
from agents.encoders import policy_input
from dungeon.environment import DungeonEnv
from .buffer import Rollout

@torch.no_grad()
def collect(policy, config, episode_seeds, rewards=None):
    envs = [DungeonEnv(config, rewards) for _ in episode_seeds]
    obs = [e.reset(s)[0] for e,s in zip(envs, episode_seeds)]
    device = next(policy.parameters()).device
    state = policy.initial_state(len(envs), device)
    previous = [None] * len(envs)
    rows = [[] for _ in range(8)]
    for _ in range(config.max_steps):
        active = torch.tensor([not e.done for e in envs], device=device)
        if not active.any():
            break
        x = torch.stack([policy_input(o,a) for o,a in zip(obs, previous)]).to(device)
        masks = torch.tensor(__import__('numpy').stack([o.action_mask() for o in obs]), device=device)
        logits, values, state = policy(x.unsqueeze(0), state)
        dist = policy.distribution(logits[0], masks)
        actions = dist.sample()
        rs, terminal = [], []
        for i,env in enumerate(envs):
            if env.done:
                rs.append(0.0)
                terminal.append(True)
            else:
                obs[i], r, term, _, _ = env.step(int(actions[i]))
                previous[i] = int(actions[i])
                rs.append(r)
                terminal.append(term)
        data = (x, masks, actions, dist.log_prob(actions), values[0],
                torch.tensor(rs, device=device), active, torch.tensor(terminal, device=device))
        for row, value in zip(rows, data):
            row.append(value)
    final_x = torch.stack([policy_input(o,a) for o,a in zip(obs,previous)]).to(device)
    _, bootstrap, _ = policy(final_x.unsqueeze(0), state)
    return Rollout(*(torch.stack(row) for row in rows), bootstrap[0], [e.metrics() for e in envs])
