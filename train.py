import argparse
from rl.trainer import train

def main():
    p = argparse.ArgumentParser(description="Train a local PPO memory policy")
    p.add_argument("--agent", choices=["no_memory","history","gru","lstm"], default="gru")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="configs/quick.json")
    p.add_argument("--output")
    p.add_argument("--updates", type=int)
    p.add_argument("--mode", choices=["procedural","showcase"])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    train(a.agent, a.seed, a.config, a.output, a.updates, a.mode, a.device)

if __name__ == "__main__":
    main()
