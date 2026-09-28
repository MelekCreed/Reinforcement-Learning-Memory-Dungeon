import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dungeon.entities import DungeonConfig
from evaluate import evaluate
from rl.trainer import seed_everything
from experiments.common import policies, save_rows, plot

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",default="results/comparison")
    p.add_argument("--maps",type=int,default=100)
    args = p.parse_args()
    seed_everything(0)
    rows = []
    for name,policy,meta in policies(args.root):
        base = meta["config"]["environment"]
        scenarios = {"unseen_same_size":{},"6x6":{"size":6,"rooms":15},"8x8":{"size":8,"rooms":20},
                     "more_keys":{"keys":4},"unreliable_clues":{"clue_reliability":.2}}
        for scenario,overrides in scenarios.items():
            cfg = DungeonConfig(**{**base,**overrides,"mode":"procedural"})
            summary, episodes = evaluate(policy,cfg,args.maps)
            rows.append({"agent":name,"scenario":scenario,**summary})
            save_rows(episodes,Path(args.root)/"generalization"/name,scenario)
    out = Path(args.root)/"generalization"
    save_rows(rows,out,"generalization")
    plot(rows,out,"generalization","scenario")
    print(rows)

if __name__ == "__main__":
    main()
