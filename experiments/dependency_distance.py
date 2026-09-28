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
    p.add_argument("--root",default="results/showcase")
    p.add_argument("--maps",type=int,default=100)
    args = p.parse_args()
    seed_everything(0)
    rows = []
    for name,policy,meta in policies(args.root):
        for distance in (2,4,8,15,25,30):
            cfg = DungeonConfig(**{**meta["config"]["environment"],"mode":"showcase","dependency":distance,"max_steps":100})
            summary, episodes = evaluate(policy,cfg,args.maps)
            # distance counts detour edges; actual cue-to-seal decision delay:
            rows.append({"agent":name,"detour_edges":distance,"dependency_decisions":2*distance+3,**summary})
            save_rows(episodes,Path(args.root)/"dependency"/name,str(distance))
    out = Path(args.root)/"dependency"
    save_rows(rows,out,"dependency_distance")
    plot(rows,out,"dependency_distance","dependency_decisions")
    print(rows)

if __name__ == "__main__":
    main()
