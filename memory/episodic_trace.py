"""External diagnostic notebook, NEVER passed to a policy."""
import networkx as nx
from dungeon.entities import COLORS, DELTAS, ROOMS, DIRECTIONS

class EpisodicTrace:
    def __init__(self):
        self.graph = nx.Graph()
        self.position = (0,0)
        self.events = []
        self.claims = []
        self._seen = set()

    def update(self, obs, step, previous_action=None, previous_obs=None):
        # Dead reckoning from observed successful moves, no true coordinates.
        if previous_action is not None and previous_action < 4 and previous_obs.passages[previous_action] == 1:
            dx,dy = DELTAS[previous_action]
            q = (self.position[0]+dx, self.position[1]+dy)
            self.graph.add_edge(self.position,q)
            self.position = q
        self.graph.add_node(self.position, room=ROOMS[obs.room], keys=list(obs.keys),
                            passages=obs.passages, seal=obs.seal, clue=obs.clue)
        def event(kind, text, key):
            if key not in self._seen:
                self._seen.add(key)
                self.events.append({"step":step,"kind":kind,"text":text,"position":self.position})
        for i,p in enumerate(obs.passages):
            if 2 <= p <= 5:
                event("door", f"{COLORS[p-2].upper()} DOOR discovered to the {DIRECTIONS[i]}", ("door",self.position,i))
        for color in obs.keys:
            event("key", f"{COLORS[color].upper()} KEY observed in {ROOMS[obs.room]}", ("key",color))
            for c,room in self.claims:
                if c == color and room != obs.room:
                    event("contradiction", f"Observed {COLORS[c]} key contradicts the note naming {ROOMS[room]}", ("contradiction",c,room))
        for color in obs.inventory:
            event("collected", f"{COLORS[color].upper()} KEY collected", ("collected",color))
        if obs.clue:
            event("clue", f"Unverified: {COLORS[obs.clue[0]]} key in {ROOMS[obs.clue[1]]}", ("clue",self.position))
            if obs.clue not in self.claims:
                self.claims.append(obs.clue)
        if obs.cue is not None:
            event("cue", f"One-time seal: {['RAVEN','MOON'][obs.cue]}", ("cue",))
        if obs.seal:
            event("exit", "Exit seals discovered", ("exit",))

    def remove_event(self, index):
        """Visual-only edit: cannot alter learned policy behavior."""
        self.events.pop(index)
