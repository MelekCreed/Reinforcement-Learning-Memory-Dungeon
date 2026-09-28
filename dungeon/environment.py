"""Small local POMDP with Gym-style reset/step, without a Gym dependency."""
from collections import Counter
import copy
from .entities import DungeonConfig, DELTAS
from .generator import generate, allowed_turnstile
from .observations import Observation
from .rewards import Rewards

class DungeonEnv:
    def __init__(self, config: DungeonConfig | None = None, rewards: Rewards | None = None):
        self.config = config or DungeonConfig()
        self.rewards = rewards or Rewards()

    def reset(self, seed: int = 0):
        self.seed = seed
        self.dungeon = generate(seed, self.config)
        self.position = self.dungeon.start
        self.inventory = set()
        self.remaining = copy.deepcopy(self.dungeon.keys)
        self.opened = set()
        self.visits = Counter({self.position: 1})
        self.path = [self.position]
        self.steps = self.invalid = self.repeats = self.backtracks = self.collected = 0
        self.total_reward = 0.0
        self.done = self.success = False
        self.first_key_step = None
        return self.observe(), self.metrics()

    def observe(self):
        passages = []
        for dx, dy in DELTAS:
            q = (self.position[0]+dx, self.position[1]+dy)
            if not self.dungeon.graph.has_edge(self.position, q):
                passages.append(0)
                continue
            c = self.dungeon.graph.edges[self.position, q].get("lock")
            if c is not None and c not in self.opened:
                passages.append(2+c)
            elif not allowed_turnstile(self.dungeon, self.position, q, bool(self.inventory)):
                passages.append(6)
            else:
                passages.append(1)
        clue = self.dungeon.clues.get(self.position)
        return Observation(self.dungeon.graph.nodes[self.position]["kind"], tuple(passages),
                           tuple(self.remaining.get(self.position, [])), tuple(sorted(self.inventory)),
                           self.dungeon.rune if self.steps == 0 else None,
                           self.position == self.dungeon.exit, clue[:2] if clue else None)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is over; call reset")
        if not 0 <= action < 13:
            raise ValueError("Unknown action")
        obs = self.observe()
        r = self.rewards.step
        self.steps += 1
        valid = obs.action_mask()[action] or action in (11, 12)
        if not valid:
            self.invalid += 1
            r += self.rewards.invalid
        elif action < 4:
            dx, dy = DELTAS[action]
            q = (self.position[0]+dx, self.position[1]+dy)
            if self.visits[q]:
                self.repeats += 1
            else:
                r += self.rewards.discovery
            # Immediate reversals without collecting/unlocking in between.
            if len(self.path) > 1 and q == self.path[-2] and getattr(self, "last_action", 12) < 4:
                self.backtracks += 1
                r += self.rewards.loop
            self.position = q
            self.visits[q] += 1
            self.path.append(q)
        elif action < 8:
            c = obs.passages[action-4]-2
            self.opened.add(c)
            r += self.rewards.unlock
        elif action == 8:
            keys = self.remaining.pop(self.position)
            self.inventory.update(keys)
            self.collected += len(keys)
            self.first_key_step = self.first_key_step or self.steps
            r += self.rewards.key * len(keys)
        elif action in (9, 10):
            self.done = True
            self.success = action-9 == self.dungeon.rune
            r += self.rewards.escape if self.success else self.rewards.wrong_seal
        terminated = self.done
        truncated = not terminated and self.steps >= self.config.max_steps
        self.done = terminated or truncated
        self.last_action = action
        self.total_reward += r
        return self.observe(), r, terminated, truncated, self.metrics()

    def metrics(self):
        return {"seed": self.seed, "success": self.success, "reward": self.total_reward,
                "steps": self.steps, "steps_to_success": self.steps if self.success else None,
                "invalid_actions": self.invalid, "repeated_visits": self.repeats,
                "immediate_backtracks": self.backtracks, "keys_collected": self.collected,
                "keys_per_step": self.collected / max(1, self.steps), "first_key_step": self.first_key_step}

    def clone(self):
        return copy.deepcopy(self)
