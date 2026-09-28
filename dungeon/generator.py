"""Seeded tree dungeons with constructive key ordering and a BFS certificate."""
import random
from collections import deque
import networkx as nx
from .entities import Dungeon, DungeonConfig, DELTAS, ROOMS

SPLITS = {"train": (0, 100_000), "validation": (100_000, 110_000), "test": (200_000, 210_000)}

def seeds(split: str, count: int, offset: int = 0) -> list[int]:
    start, end = SPLITS[split]
    if offset < 0 or count < 1 or start + offset + count > end:
        raise ValueError("Requested seeds exceed the split")
    return list(range(start + offset, start + offset + count))

def generate(seed: int, cfg: DungeonConfig) -> Dungeon:
    rng = random.Random(seed)
    if cfg.mode == "showcase":
        # Snake on a grid. Start sees a door, follows turnstiles to the key,
        # then returns; a vanishing rune cue must survive the entire detour.
        route = [(x if y % 2 == 0 else 7-x, y) for y in range(5) for x in range(8)][:cfg.dependency + 2]
        g = nx.path_graph(route)
        d = Dungeon(g, route[1], route[0], rng.randrange(2), {route[-1]: [0]}, mode="showcase", route=route)
        g.edges[route[0], route[1]]["lock"] = 0
    else:
        start = (rng.randrange(cfg.size), rng.randrange(cfg.size))
        g = nx.Graph()
        g.add_node(start)
        while len(g) < cfg.rooms:
            frontier = [(p, (p[0]+dx, p[1]+dy)) for p in g for dx, dy in DELTAS
                        if 0 <= p[0]+dx < cfg.size and 0 <= p[1]+dy < cfg.size
                        and (p[0]+dx, p[1]+dy) not in g]
            a, b = rng.choice(frontier)
            g.add_edge(a, b)
        distances = nx.single_source_shortest_path_length(g, start)
        exit_room = max(distances, key=distances.get)
        path = nx.shortest_path(g, start, exit_room)
        # All locks lie on the unique exit path. Each key is reachable before
        # its own lock; later keys may require earlier keys, never themselves.
        nlocks = min(cfg.keys, len(path)-1)
        if nlocks < cfg.keys:
            return generate(seed + 1_000_000_007, cfg)
        indices = sorted(rng.sample(range(len(path)-1), nlocks))
        colors = rng.sample(range(4), nlocks)
        for i, color in zip(indices, colors):
            g.edges[path[i], path[i+1]]["lock"] = color
        d = Dungeon(g, start, exit_room, rng.randrange(2))
        for i, color in zip(indices, colors):
            cut = g.copy()
            cut.remove_edge(path[i], path[i+1])
            candidates = sorted(nx.node_connected_component(cut, start))
            p = rng.choice(candidates)
            d.keys.setdefault(p, []).append(color)
    for p in d.graph:
        d.graph.nodes[p]["kind"] = rng.randrange(len(ROOMS))
    for p in d.graph:
        if rng.random() < 0.45:
            target = rng.choice(list(d.keys))
            color = rng.choice(d.keys[target])
            truth = rng.random() < cfg.clue_reliability
            kind = d.graph.nodes[target]["kind"]
            claimed = kind if truth else rng.choice([k for k in range(len(ROOMS)) if k != kind])
            d.clues[p] = (color, claimed, truth)
    return d

def solution(d: Dungeon) -> list[int]:
    """Privileged shortest action plan, strictly for validation/oracle baseline."""
    from .entities import direction
    # Locks need explicit unlock actions. Inventory is monotonic; store opened colors.
    initial = (d.start, 0, 0)
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        (p, bag, opened), plan = queue.popleft()
        if p == d.exit:
            return plan + [9 + d.rune]
        nexts = []
        available = d.keys.get(p, [])
        newbag = bag
        for c in available:
            newbag |= 1 << c
        if newbag != bag:
            nexts.append(((p, newbag, opened), 8))
        for q in d.graph[p]:
            c = d.graph.edges[p, q].get("lock")
            di = direction(p, q)
            if c is not None and not opened & (1 << c):
                if bag & (1 << c):
                    nexts.append(((p, bag, opened | (1 << c)), 4 + di))
            elif d.mode != "showcase" or allowed_turnstile(d, p, q, bool(bag)):
                nexts.append(((q, bag, opened), di))
        for state, action in nexts:
            if state not in seen:
                seen.add(state)
                queue.append((state, plan + [action]))
    raise ValueError("Unsolvable dungeon")

def allowed_turnstile(d, p, q, has_key):
    if d.mode != "showcase":
        return True
    i, j = d.route.index(p), d.route.index(q)
    return j < i if has_key else j > i
