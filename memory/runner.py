from dataclasses import asdict
import copy
import torch
from agents.base import clone_state
from agents.encoders import policy_input
from dungeon.environment import DungeonEnv
from dungeon.entities import ACTIONS
from .episodic_trace import EpisodicTrace
from .interventions import intervene, vector

class Session:
    def __init__(self, policy, config, seed):
        self.policy = policy
        self.env = DungeonEnv(config)
        self.obs, _ = self.env.reset(seed)
        self.state = policy.initial_state(1)
        self.previous_action = None
        self.trace = EpisodicTrace()
        self.trace.update(self.obs, 0)
        self.timeline = []
        self.snapshots = [clone_state(self.state)]
        self.interventions = []

    @torch.no_grad()
    def step(self, action=None):
        if self.env.done:
            return
        x = policy_input(self.obs, self.previous_action).to(next(self.policy.parameters()).device)
        logits, _, self.state = self.policy(x[None,None], self.state)
        mask = torch.as_tensor(self.obs.action_mask(), device=x.device)
        dist = self.policy.distribution(logits[0,0], mask)
        chosen = int(dist.probs.argmax()) if action is None else action
        old = self.obs
        self.obs, reward, _, _, _ = self.env.step(chosen)
        self.previous_action = chosen
        self.trace.update(self.obs, self.env.steps, chosen, old)
        self.snapshots.append(clone_state(self.state))
        self.timeline.append({"step":self.env.steps, "action":ACTIONS[chosen], "action_id":chosen,
                              "reward":reward, "position":list(self.env.position),
                              "observation":old.text(), "hidden":vector(self.state),
                              "probabilities":dist.probs.cpu().tolist()})

    def apply(self, kind, **kwargs):
        before = vector(self.state)
        self.state = intervene(self.state, kind, **kwargs)
        self.interventions.append({"step":self.env.steps, "kind":kind,
                                   "before":before, "after":vector(self.state)})

    def fork(self):
        # Sharing immutable policy weights is safe, never duplicate the network.
        return copy.deepcopy(self, {id(self.policy): self.policy})

    def finish(self):
        while not self.env.done:
            self.step()
        return self

    def record(self):
        return {"metrics":self.env.metrics(), "config":asdict(self.env.config),
                "timeline":self.timeline, "interventions":self.interventions,
                "events":self.trace.events}

def paired(policy, config, seed, at_step, kind="reset", **kwargs):
    session = Session(policy,config,seed)
    while session.env.steps < at_step and not session.env.done:
        session.step()
    if session.env.done:
        raise ValueError("Intervention must precede episode termination")
    control, treatment = session.fork(), session.fork()
    treatment.apply(kind, **kwargs)
    return control.finish(), treatment.finish()
