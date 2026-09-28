from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from agents.base import Policy, PolicyConfig
from dungeon.entities import DungeonConfig
from dungeon.generator import seeds
from dungeon.rewards import Rewards
from .ppo import PPOConfig, update
from .rollout import collect

ROOT = Path(__file__).resolve().parents[1]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def train(agent="gru", seed=42, config_path=None, output=None, updates=None, mode=None, device="cpu"):
    cfg = json.loads(Path(config_path or ROOT/"configs/quick.json").read_text())
    if updates is not None:
        cfg["updates"] = updates
    if mode:
        cfg["environment"]["mode"] = mode
    seed_everything(seed)
    policy = Policy(PolicyConfig(agent=agent, **cfg.get("policy", {}))).to(device)
    ppo = PPOConfig(**cfg.get("ppo", {}))
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate, eps=1e-5)
    output = Path(output or ROOT/"results"/agent)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"seed": seed, "config": cfg, "agent": agent, "torch": str(torch.__version__),
                "numpy": str(np.__version__), "device": device, "training_split": "train"}
    (output/"config.json").write_text(json.dumps(metadata, indent=2))
    logs = []
    env_cfg = DungeonConfig(**cfg["environment"])
    rewards = Rewards(**cfg.get("rewards", {}))
    seed_pool = seeds("train", 100_000)
    rng = random.Random(seed)
    for iteration in range(cfg["updates"]):
        current = env_cfg
        curriculum = cfg.get("curriculum", [])
        if curriculum and env_cfg.mode == "showcase":
            distance = curriculum[min(len(curriculum)-1, iteration*len(curriculum)//cfg["updates"])]
            current = DungeonConfig(**{**asdict(env_cfg), "dependency": distance})
        batch = collect(policy, current, rng.sample(seed_pool, cfg["batch_size"]), rewards)
        stats = update(policy, optimizer, batch, ppo)
        stats.update(update=iteration+1, timesteps=sum(m["steps"] for m in batch.metrics),
                     reward=float(np.mean([m["reward"] for m in batch.metrics])),
                     success=float(np.mean([m["success"] for m in batch.metrics])), dependency=current.dependency)
        logs.append(stats)
        if (iteration+1) % 25 == 0 or iteration == 0:
            print(f"{agent} update {iteration+1}/{cfg['updates']} reward={stats['reward']:.2f} success={stats['success']:.1%}", flush=True)
        if (iteration+1) % 100 == 0:
            policy.save(output/"checkpoint.pt", metadata)
            (output/"training.json").write_text(json.dumps(logs, indent=2))
    policy.save(output/"checkpoint.pt", metadata)
    (output/"training.json").write_text(json.dumps(logs, indent=2))
    import pandas as pd
    pd.DataFrame(logs).to_csv(output/"training.csv", index=False)
    return policy, metadata
