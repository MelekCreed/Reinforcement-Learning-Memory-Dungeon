import numpy as np
import pytest
from dungeon.environment import DungeonEnv
from dungeon.entities import DungeonConfig
from dungeon.generator import generate, solution, seeds
from dungeon.observations import FEATURE_DIM

@pytest.mark.parametrize("size,rooms,keys", [(5,8,1),(6,15,3),(8,20,4)])
def test_solvable(size, rooms, keys):
    for seed in range(40):
        cfg = DungeonConfig(size=size, rooms=rooms, keys=keys, max_steps=300)
        env = DungeonEnv(cfg)
        env.reset(seed)
        plan = solution(env.dungeon)
        for action in plan:
            env.step(action)
        assert env.success and env.invalid == 0
        assert len(env.dungeon.graph) == rooms
        assert sum(len(v) for v in env.dungeon.keys.values()) == keys

def test_locked_doors_require_key():
    env = DungeonEnv(DungeonConfig(mode="showcase"))
    obs, _ = env.reset(1)
    lock = next(i for i,p in enumerate(obs.passages) if p == 2)
    position = env.position
    env.step(lock)
    env.step(lock+4)
    assert env.position == position and not env.opened and env.invalid == 2

def test_observation_isolation_and_one_time_cue():
    env = DungeonEnv()
    obs, _ = env.reset(42)
    assert set(vars(obs)) == {"room","passages","keys","inventory","cue","seal","clue"}
    assert obs.features().shape == (FEATURE_DIM,)
    assert obs.cue in (0,1)
    obs2, *_ = env.step(11)
    assert obs2.cue is None
    assert "coordinate" not in obs.text()
    old = obs2.features().copy()
    env.dungeon.rune = 1-env.dungeon.rune
    np.testing.assert_array_equal(old, env.observe().features())

def test_seed_reproduction():
    a, b = generate(123, DungeonConfig()), generate(123, DungeonConfig())
    assert list(a.graph.edges(data=True)) == list(b.graph.edges(data=True))
    assert a.keys == b.keys and a.clues == b.clues and a.rune == b.rune

def test_done_and_timeout():
    env = DungeonEnv(DungeonConfig(max_steps=1))
    env.reset()
    _, _, term, trunc, _ = env.step(12)
    assert trunc and not term
    with pytest.raises(RuntimeError):
        env.step(12)

def test_splits():
    sets = [set(seeds(s, 1000)) for s in ("train","validation","test")]
    assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])
    with pytest.raises(ValueError):
        seeds("test", 10001)

def test_showcase_oracle():
    for distance in (2,8,15,30):
        env = DungeonEnv(DungeonConfig(mode="showcase", dependency=distance))
        env.reset(1)
        for a in solution(env.dungeon):
            env.step(a)
        assert env.success
