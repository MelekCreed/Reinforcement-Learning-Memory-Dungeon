"""The sole policy interface. Contains no coordinates, IDs, truth labels or map."""
from dataclasses import dataclass
import numpy as np
from .entities import COLORS, DIRECTIONS, ROOMS

FEATURE_DIM = 6 + 4*7 + 4 + 4 + 3 + 1 + 5 + 7

@dataclass(frozen=True)
class Observation:
    room: int
    passages: tuple[int, ...]  # 0 wall, 1 corridor, 2..5 lock, 6 blocked turnstile
    keys: tuple[int, ...]
    inventory: tuple[int, ...]
    cue: int | None
    seal: bool
    clue: tuple[int, int] | None

    def features(self) -> np.ndarray:
        def oh(i, n):
            a = np.zeros(n, dtype=np.float32)
            a[i] = 1
            return a
        key = np.zeros(4, dtype=np.float32)
        bag = key.copy()
        key[list(self.keys)] = 1
        bag[list(self.inventory)] = 1
        return np.concatenate([oh(self.room, 6), *(oh(p, 7) for p in self.passages), key, bag,
                               oh(0 if self.cue is None else self.cue+1, 3), [float(self.seal)],
                               oh(0 if self.clue is None else self.clue[0]+1, 5),
                               oh(0 if self.clue is None else self.clue[1]+1, 7)]).astype(np.float32)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(13, dtype=bool)
        for i, p in enumerate(self.passages):
            mask[i] = p == 1
            mask[4+i] = 2 <= p <= 5 and p-2 in self.inventory
        mask[8] = bool(self.keys)
        mask[9:11] = self.seal
        # Inspect/wait remain legal in human play. Learned policies mask these
        # no-information actions in every architecture, explicitly documented.
        if not mask.any():
            mask[12] = True
        return mask

    def text(self) -> str:
        lines = [f"You are in a {ROOMS[self.room]}.", "", "Visible passages:"]
        for di, p in enumerate(self.passages):
            if p:
                label = "open corridor" if p == 1 else "closed one-way turnstile" if p == 6 else f"locked {COLORS[p-2]} door"
                lines.append(f"• {DIRECTIONS[di]}: {label}")
        lines += ["", "Objects: " + (", ".join(COLORS[k]+" key" for k in self.keys) or "none"),
                  "Inventory: " + (", ".join(COLORS[k]+" key" for k in self.inventory) or "empty")]
        if self.clue:
            lines += [f"An unverified note says: the {COLORS[self.clue[0]]} key is in a {ROOMS[self.clue[1]]}."]
        if self.cue is not None:
            lines += [f"A dying torch reveals {'RAVEN' if self.cue == 0 else 'MOON'}. Remember this seal. The inscription will vanish after your first action."]
        if self.seal:
            lines += ["The exit offers two seals: RAVEN and MOON. Invoke the remembered seal. A wrong choice ends the expedition."]
        return "\n".join(lines)
