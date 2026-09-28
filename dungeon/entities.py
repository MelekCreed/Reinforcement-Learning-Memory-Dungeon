from dataclasses import dataclass, field
import networkx as nx

DIRECTIONS = ("north", "south", "east", "west")
DELTAS = ((0, -1), (0, 1), (1, 0), (-1, 0))
COLORS = ("red", "blue", "brass", "violet")
ROOMS = ("chamber", "library", "armory", "flooded room", "crypt", "observatory")
ACTIONS = tuple(f"move {d}" for d in DIRECTIONS) + tuple(f"unlock {d}" for d in DIRECTIONS) + ("take key", "invoke raven", "invoke moon", "inspect room", "wait")

@dataclass(frozen=True)
class DungeonConfig:
    size: int = 5
    rooms: int = 12
    keys: int = 2
    clue_reliability: float = 0.75
    max_steps: int = 160
    mode: str = "procedural"
    dependency: int = 8

    def __post_init__(self):
        if not 5 <= self.size <= 16 or not 4 <= self.rooms <= self.size**2:
            raise ValueError("Use size 5..16 and rooms 4..size²")
        if not 1 <= self.keys <= min(4, self.rooms - 2):
            raise ValueError("Use 1..4 keys, fewer than rooms - 1")
        if not 0 <= self.clue_reliability <= 1 or self.max_steps < 1:
            raise ValueError("Invalid reliability or time limit")
        if self.mode not in ("procedural", "showcase") or not 2 <= self.dependency <= 30:
            raise ValueError("Invalid mode or dependency (2..30)")

@dataclass
class Dungeon:
    graph: nx.Graph
    start: tuple[int, int]
    exit: tuple[int, int]
    rune: int
    keys: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    clues: dict = field(default_factory=dict)
    mode: str = "procedural"
    route: list = field(default_factory=list)

def direction(a, b):
    return DELTAS.index((b[0] - a[0], b[1] - a[1]))
