"""Change ONLY the initial cue and required seal; keep nuisance observations fixed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from agents.base import load_policy
from dungeon.entities import DungeonConfig
from dungeon.generator import seeds
from memory.runner import Session
from memory.episodic_trace import EpisodicTrace
from rl.trainer import seed_everything
from experiments.common import save_rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="results/showcase/gru/checkpoint.pt")
    p.add_argument("--maps",type=int,default=30)
    args = p.parse_args()
    seed_everything(0)
    policy,meta = load_policy(args.checkpoint)
    cfg = DungeonConfig(**meta["config"]["environment"])
    if cfg.mode != "showcase":
        raise ValueError("Counterfactual audit is defined for the controlled showcase")
    rows = []
    for seed in seeds("test",args.maps,offset=1000):
        a = Session(policy,cfg,seed)
        b = a.fork()
        b.env.dungeon.rune = 1-b.env.dungeon.rune
        b.obs = b.env.observe()
        b.trace = EpisodicTrace()
        b.trace.update(b.obs,0)
        a.finish()
        b.finish()
        rows.append({"seed":seed,"original_success":a.env.success,"flipped_success":b.env.success,
                     "same_navigation": [r['action_id'] for r in a.timeline[:-1]] == [r['action_id'] for r in b.timeline[:-1]],
                     "opposite_choice":a.timeline[-1]['action_id'] != b.timeline[-1]['action_id']})
    save_rows(rows,ROOT/"results/ablation","cue_counterfactual")
    print(f"Both cue versions solved: {sum(r['original_success'] and r['flipped_success'] for r in rows)}/{len(rows)}")

if __name__ == "__main__":
    main()
