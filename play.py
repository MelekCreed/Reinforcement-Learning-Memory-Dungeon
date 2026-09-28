"""Human-playable text dungeon: python play.py --seed 42."""
import argparse
from dungeon.environment import DungeonEnv
from dungeon.entities import ACTIONS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    env = DungeonEnv()
    obs, _ = env.reset(args.seed)
    print("Actions: " + ", ".join(ACTIONS))
    while not env.done:
        print(obs.text())
        command = input("> ").strip().lower()
        if command in ("quit", "exit"):
            return
        if command not in ACTIONS:
            print("Unknown action")
            continue
        obs, reward, _, _, _ = env.step(ACTIONS.index(command))
        print(f"Reward {reward:+.2f}")
    print(env.metrics())

if __name__ == "__main__":
    main()
