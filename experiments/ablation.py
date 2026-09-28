import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from agents.base import load_policy
from dungeon.entities import DungeonConfig
from dungeon.generator import seeds
from memory.runner import Session, paired
from memory.visualization import hidden_figure, trajectory_figure
from rl.trainer import seed_everything
from evaluate import summarize
from experiments.common import save_rows, plot, paired_interval

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="results/showcase/gru/checkpoint.pt")
    p.add_argument("--maps",type=int,default=100)
    p.add_argument("--output",default="results/ablation")
    a = p.parse_args()
    seed_everything(0)
    policy,meta = load_policy(a.checkpoint)
    if policy.config.agent not in ("gru","lstm"):
        raise ValueError("This experiment requires a recurrent policy")
    cfg = DungeonConfig(**meta["config"]["environment"])
    controls = {s:Session(policy,cfg,s).finish() for s in seeds("test",a.maps)}
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    rows, pairs = [], []
    showcase = None
    for kind, strengths in (("reset",(.25,.5,.75)),("noise",(0.,.1,.5,1.,2.))):
        for strength in strengths:
            results, deltas = [], []
            for seed,original in controls.items():
                # Fraction of the CONTROL trajectory length, determined before treatment.
                at = max(1,min(original.env.steps-1,int(original.env.steps*(strength if kind == "reset" else .5))))
                control,treatment = paired(policy,cfg,seed,at,kind,noise=strength,seed=seed)
                results.append(treatment.env.metrics())
                delta = int(treatment.env.success)-int(control.env.success)
                deltas.append(delta)
                pairs.append({"kind":kind,"strength":strength,"seed":seed,"at_step":at,
                              "control_success":control.env.success,"treatment_success":treatment.env.success,
                              "control_steps":control.env.steps,"treatment_steps":treatment.env.steps})
                if showcase is None and kind == "reset" and control.env.success and not treatment.env.success:
                    showcase = {"seed":seed,"at_step":at,"control":control.record(),"treatment":treatment.record()}
            rows.append({"agent":kind,"strength":strength,**summarize(results),
                         "control_success_rate":sum(s.env.success for s in controls.values())/len(controls),
                         "success_delta":sum(deltas)/len(deltas),"paired_delta_ci95":paired_interval(deltas)})
    save_rows(rows,out,"ablation")
    save_rows(pairs,out,"paired_outcomes")
    for kind in ("reset","noise"):
        plot([r for r in rows if r["agent"] == kind],out,"memory_"+kind,"strength")
    if showcase:
        (out/"showcase_episode.json").write_text(json.dumps(showcase,indent=2))
        for name in ("control","treatment"):
            fig = hidden_figure(showcase[name]["timeline"])
            fig.savefig(out/f"hidden_{name}.png",dpi=160)
            fig = trajectory_figure(showcase[name]["timeline"])
            fig.savefig(out/f"trajectory_{name}.png",dpi=160)
        print(f"Verified deterministic memory deletion failure: seed={showcase['seed']} step={showcase['at_step']}")
    else:
        print("No successful-control/failed-treatment pair found; no causal success claim is warranted.")
    print(rows)

if __name__ == "__main__":
    main()
