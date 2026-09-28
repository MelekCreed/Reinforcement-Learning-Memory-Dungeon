"""Train all three variants on identical seed streams, evaluate held-out maps."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dungeon.entities import DungeonConfig
from rl.trainer import train, seed_everything
from evaluate import evaluate
from experiments.common import policies, save_rows, plot, training_plots

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",default="configs/quick.json")
    p.add_argument("--root",default="results/comparison")
    p.add_argument("--maps",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--skip-training",action="store_true")
    args = p.parse_args()
    if not args.skip_training:
        for agent in ("no_memory","history","gru"):
            train(agent,args.seed,args.config,Path(args.root)/agent)
    seed_everything(args.seed)
    rows = []
    for name,policy,meta in policies(args.root):
        for split in ("validation","test"):
            summary, episodes = evaluate(policy,DungeonConfig(**meta["config"]["environment"]),args.maps,split)
            rows.append({"agent":name,"split":split,**summary})
            save_rows(episodes,Path(args.root)/name,split+"_episodes")
    save_rows(rows,args.root,"comparison")
    plot(rows,args.root,"success_by_architecture","split")
    plot(rows,args.root,"steps_to_exit","split","steps_to_success")
    training_plots(args.root)
    print(rows)

if __name__ == "__main__":
    main()
