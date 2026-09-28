"""Regenerate the report and checkpoint provenance from actual output files."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import hashlib

def read(relative):
    path = ROOT/relative
    return json.loads(path.read_text()) if path.exists() else []

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])

def main():
    text = ["# Recorded experiment results", "", "Generated from saved JSON measurements by `python experiments/report.py`. No manually entered scores.",
            "", "All runs use one optimization seed (42), CPU PyTorch, shared map seed streams, and greedy evaluation. Confidence intervals quantify map sampling only. Showcase and procedural models are trained separately."]
    for family,desc in (("showcase","Controlled cue retention: 800 PPO updates, 32 episodes/update, detour curriculum 2/4/8/15."),
                        ("comparison","Quick procedural exploration: 100 PPO updates, 16 episodes/update, 5×5 / 8 rooms / 1 key. Undertrained.")):
        rows = read(f"results/{family}/comparison.json")
        text += ["",f"## {family.title()}","",desc,""]
        if not rows:
            text += ["No comparison file available."]
            continue
        text += [table(["Agent","Split","Maps","Success","95% Wilson CI","Mean reward","Steps / success"],
                       [[r['agent'],r['split'],r['maps'],f"{r['success_rate']:.0%}",f"{r['success_ci95_low']:.1%}–{r['success_ci95_high']:.1%}",f"{r['mean_reward']:.2f}","—" if r['steps_to_success'] is None else f"{r['steps_to_success']:.1f}"] for r in rows])]
    rows = read("results/ablation/ablation.json")
    text += ["","## Paired interventions","","Reset timing is a fraction of the control trajectory length. Noise is per-dimension Gaussian standard deviation, applied at 50%. Positive control: zero noise must preserve behavior.","",
             table(["Intervention","Strength / fraction","Control","Treatment","Success change","Paired 95% bootstrap CI"],
                   [[r['agent'],r['strength'],f"{r['control_success_rate']:.0%}",f"{r['success_rate']:.0%}",f"{r['success_delta']:+.0%}",f"{r['paired_delta_ci95'][0]:+.1%} to {r['paired_delta_ci95'][1]:+.1%}"] for r in rows])]
    example = read("results/ablation/showcase_episode.json")
    if example:
        a,b = example['control'],example['treatment']
        first = next(i+1 for i,(x,y) in enumerate(zip(a['timeline'],b['timeline'])) if x['action_id'] != y['action_id'])
        text += ["",f"Verified replay: seed **{example['seed']}**, reset after **{example['at_step']}** decisions, first action divergence at **{first}**. Control success **{a['metrics']['success']}**; treatment success **{b['metrics']['success']}**. This example was selected after inspecting the test pairs to illustrate an effect; the aggregate uses every tested seed."]
    rows = read("results/comparison/generalization/generalization.json")
    twins = read("results/ablation/cue_counterfactual.json")
    if twins:
        solved = sum(r['original_success'] and r['flipped_success'] for r in twins)
        opposite = sum(r['opposite_choice'] for r in twins)
        text += ["","## Cue counterfactual audit","",f"On **{solved}/{len(twins)}** additional held-out pairs, the GRU solved both versions of the same dungeon when only the original cue and required seal were flipped. It changed its final choice in **{opposite}/{len(twins)}** pairs. Navigation, room descriptions and clues were held fixed. These pairs use test seeds starting at 201000, separate from the main 100-map evaluation."]
    scenarios = list(dict.fromkeys(r['scenario'] for r in rows))
    agents = list(dict.fromkeys(r['agent'] for r in rows))
    text += ["","## Procedural generalization","","These weak quick-run results do not support a generalization-performance claim.","",
             table(["Agent"]+scenarios,[[agent]+[f"{next(r['success_rate'] for r in rows if r['agent']==agent and r['scenario']==s):.0%}" for s in scenarios] for agent in agents])]
    rows = read("results/showcase/dependency/dependency_distance.json")
    if rows:
        distances = sorted(set(r['dependency_decisions'] for r in rows))
        agents = list(dict.fromkeys(r['agent'] for r in rows))
        text += ["","## Cue-to-decision dependency distance","","Longer detours use the same controlled corridor family, not new free-navigation topologies. Every tested delay exceeds the history-4 window, so this sweep does not locate its performance boundary.","",
                 table(["Agent"]+[str(d)+" decisions" for d in distances],[[a]+[f"{next(r['success_rate'] for r in rows if r['agent']==a and r['dependency_decisions']==d):.0%}" for d in distances] for a in agents])]
    text += ["","## Local checkpoint provenance","","Weights are excluded from Git. These hashes identify the locally evaluated checkpoints; retraining under a different software/hardware stack may not reproduce identical bytes.",""]
    manifest = []
    for p in sorted((ROOT/"results").glob("*/*/checkpoint.pt")):
        if "smoke" not in p.parts:
            manifest.append({"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size})
    text += [table(["Checkpoint","SHA-256","Bytes"],[[r['path'],r['sha256'],r['bytes']] for r in manifest]),""]
    (ROOT/"docs/RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    (ROOT/"results/checkpoint_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Generated docs/RESULTS.md and checkpoint manifest")

if __name__ == "__main__":
    main()
