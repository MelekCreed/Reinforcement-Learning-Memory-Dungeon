import torch
from agents.base import Policy, PolicyConfig
from dungeon.entities import DungeonConfig
from memory.interventions import intervene
from memory.runner import Session, paired
from rl.trainer import seed_everything

def test_interventions_are_independent():
    h = torch.ones(1,1,16)
    assert intervene(h).count_nonzero() == 0
    assert h.count_nonzero() == 16
    p = intervene(h,"partial")
    assert p[...,:8].count_nonzero() == 0 and p[...,8:].sum() == 8
    assert torch.equal(intervene(h,"noise",seed=4),intervene(h,"noise",seed=4))
    assert torch.equal(intervene(h,"restore",earlier=h*3),h*3)
    assert all(x.count_nonzero() == 0 for x in intervene((h,h)))

def test_clone_does_not_share_environment_or_state():
    s = Session(Policy(PolicyConfig()),DungeonConfig(mode="showcase"),2)
    s.step()
    f = s.fork()
    f.step()
    f.apply("reset")
    assert s.env.steps == 1 and f.env.steps == 2
    assert s.state.count_nonzero() > 0

def test_paired_prefix_and_trace_isolation():
    seed_everything(2)
    cfg = DungeonConfig(mode="showcase",dependency=4)
    policy = Policy(PolicyConfig())
    a,b = paired(policy,cfg,5,4)
    assert a.timeline[:4] == b.timeline[:4]
    assert a.env.seed == b.env.seed
    assert len(b.interventions) == 1
    assert len(a.trace.graph) <= len(a.env.dungeon.graph)
    state = a.state.clone()
    a.trace.remove_event(0)
    torch.testing.assert_close(state,a.state)
