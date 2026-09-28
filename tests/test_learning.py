import torch
import pytest
from agents.base import Policy, PolicyConfig
from agents.encoders import INPUT_DIM, policy_input
from dungeon.environment import DungeonEnv
from dungeon.entities import DungeonConfig
from rl.rollout import collect
from rl.ppo import PPOConfig, update
from rl.trainer import seed_everything

@pytest.mark.parametrize("kind", ["no_memory","history","gru","lstm"])
def test_online_matches_sequence(kind):
    seed_everything(4)
    model = Policy(PolicyConfig(kind, hidden=16))
    x = torch.randn(8,2,INPUT_DIM)
    expected, _, _ = model(x)
    state = None
    outputs = []
    for row in x:
        out, _, state = model(row[None], state)
        outputs.append(out)
    torch.testing.assert_close(torch.cat(outputs), expected, atol=1e-6, rtol=1e-5)

def test_recurrent_gradient_reaches_old_observation():
    model = Policy(PolicyConfig("gru", hidden=16))
    x = torch.randn(20,1,INPUT_DIM, requires_grad=True)
    logits, _, _ = model(x)
    logits[-1].sum().backward()
    assert x.grad[0].abs().sum() > 0

def test_ppo_updates_and_is_finite():
    seed_everything(4)
    policy = Policy(PolicyConfig("gru", hidden=16))
    cfg = DungeonConfig(mode="showcase", dependency=2, max_steps=20)
    batch = collect(policy, cfg, [1,2,3,4])
    before = policy.actor.weight.detach().clone()
    stats = update(policy, torch.optim.Adam(policy.parameters(), lr=.001), batch, PPOConfig(epochs=1))
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert not torch.equal(before, policy.actor.weight)

def test_no_memory_ignores_previous_action():
    model = Policy(PolicyConfig("no_memory"))
    obs, _ = DungeonEnv().reset(42)
    a, _, _ = model(policy_input(obs, 0)[None,None])
    b, _, _ = model(policy_input(obs, 1)[None,None])
    torch.testing.assert_close(a,b)
