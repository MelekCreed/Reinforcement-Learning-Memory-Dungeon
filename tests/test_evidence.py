"""Validate the recorded causal example when its local checkpoint is available."""
from pathlib import Path
import json
import pytest
from agents.base import load_policy
from dungeon.entities import DungeonConfig
from memory.runner import paired
from rl.trainer import seed_everything

def test_recorded_showcase_replays():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root/"results/showcase/gru/checkpoint.pt"
    evidence = root/"results/ablation/showcase_episode.json"
    if not checkpoint.exists() or not evidence.exists():
        pytest.skip("Train a showcase GRU and run ablations to verify local evidence")
    seed_everything(0)
    record = json.loads(evidence.read_text())
    model,_ = load_policy(checkpoint)
    cfg = DungeonConfig(**record["control"]["config"])
    control,treatment = paired(model,cfg,record["seed"],record["at_step"])
    assert control.env.success and not treatment.env.success
    assert control.timeline[:record["at_step"]] == treatment.timeline[:record["at_step"]]
    assert [t['action_id'] for t in control.timeline] == [t['action_id'] for t in record['control']['timeline']]
