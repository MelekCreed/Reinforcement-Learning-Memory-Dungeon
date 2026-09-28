import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from agents.base import load_policy
from dungeon.entities import DungeonConfig
from dungeon.generator import seeds
from memory.runner import Session
from rl.trainer import seed_everything

def summarize(rows):
    successes = [r for r in rows if r["success"]]
    n = len(rows)
    p = len(successes)/n
    z = 1.96
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"maps":n,"success_rate":p,"success_ci95_low":center-half,"success_ci95_high":center+half,
            "mean_reward":float(np.mean([r["reward"] for r in rows])),
            "steps_to_success":float(np.mean([r["steps"] for r in successes])) if successes else None,
            **{f"mean_{k}":float(np.mean([r[k] for r in rows])) for k in ("invalid_actions","repeated_visits","immediate_backtracks","keys_per_step")}}

def evaluate(policy, cfg, maps=100, split="test", offset=0):
    rows = [Session(policy,cfg,s).finish().env.metrics() for s in seeds(split,maps,offset)]
    return summarize(rows), rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--maps", type=int, default=100)
    p.add_argument("--split", choices=["train","validation","test"], default="test")
    p.add_argument("--output", default="results/evaluation")
    p.add_argument("--mode", choices=["procedural","showcase"])
    a = p.parse_args()
    seed_everything(0)
    policy, meta = load_policy(a.checkpoint)
    cfg = dict(meta["config"]["environment"])
    if a.mode:
        cfg["mode"] = a.mode
    cfg = DungeonConfig(**cfg)
    summary, rows = evaluate(policy,cfg,a.maps,a.split)
    output = Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps({"checkpoint":str(Path(a.checkpoint).resolve()),"environment":asdict(cfg),"split":a.split,"summary":summary,"episodes":rows},indent=2))
    import pandas as pd
    pd.DataFrame(rows).to_csv(output/"episodes.csv", index=False)
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
