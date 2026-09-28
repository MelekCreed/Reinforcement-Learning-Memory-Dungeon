import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agents.base import load_policy
from memory.visualization import style

def policies(root):
    root = Path(root)
    found = []
    for name in ("no_memory","history","gru","lstm"):
        path = root/name/"checkpoint.pt"
        if path.exists():
            policy, meta = load_policy(path)
            found.append((name,policy,meta))
    if not found:
        raise FileNotFoundError(f"No checkpoints in {root}; run compare_memory first")
    return found

def save_rows(rows, output, name):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/f"{name}.json").write_text(json.dumps(rows,indent=2))
    pd.DataFrame(rows).to_csv(output/f"{name}.csv",index=False)

def plot(rows, output, name, x, y="success_rate", title=None):
    fig,ax = plt.subplots(figsize=(8,4.5))
    style(ax)
    df = pd.DataFrame(rows)
    for agent, group in df.groupby("agent",sort=False):
        ax.plot(group[x].astype(str), group[y], marker="o",label=agent)
    ax.set(xlabel=x.replace("_"," "),ylabel=y.replace("_"," "),title=title or name.replace("_"," "))
    if y == "success_rate":
        ax.set_ylim(-.03,1.05)
    ax.legend(facecolor="#263c50",labelcolor="white")
    ax.grid(alpha=.12)
    fig.tight_layout()
    fig.savefig(Path(output)/f"{name}.png",dpi=160)
    plt.close(fig)

def training_plots(root):
    root = Path(root)
    rows = []
    for path in root.glob("*/training.json"):
        rows.extend({**r,"agent":path.parent.name} for r in json.loads(path.read_text()))
    if not rows:
        return
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax in axes:
        style(ax)
    df = pd.DataFrame(rows)
    for name,g in df.groupby("agent"):
        for ax,col in zip(axes,["reward","success"]):
            ax.plot(g["update"],g[col].rolling(20,min_periods=1).mean(),label=name)
            ax.set(xlabel="PPO update",ylabel=col,title=f"Training {col} · rolling 20")
    axes[0].legend(facecolor="#263c50",labelcolor="white")
    fig.tight_layout()
    fig.savefig(root/"training_curves.png",dpi=160)
    plt.close(fig)

def paired_interval(deltas):
    rng = np.random.default_rng(0)
    a = np.asarray(deltas,dtype=float)
    means = rng.choice(a,size=(2000,len(a)),replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]
