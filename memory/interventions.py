import torch
from agents.base import clone_state

def intervene(state, kind="reset", fraction=0.5, noise=0.5, seed=0, earlier=None):
    """Modify an independent state copy. LSTM interventions affect h AND c."""
    if kind == "restore":
        if earlier is None:
            raise ValueError("Restore requires an earlier snapshot")
        return clone_state(earlier)
    if kind not in ("reset", "partial", "noise"):
        raise ValueError("Unknown intervention")
    if not 0 <= fraction <= 1 or noise < 0:
        raise ValueError("Invalid intervention strength")
    if state is None:
        return None
    generator = torch.Generator(device=(state[0] if isinstance(state,tuple) else state).device).manual_seed(seed)
    def edit(x):
        out = x.detach().clone()
        if kind == "reset":
            out.zero_()
        elif kind == "partial":
            out[..., :int(out.shape[-1]*fraction)] = 0
        elif kind == "noise":
            out += noise*torch.randn(out.shape, generator=generator, device=out.device)
        return out
    return tuple(edit(x) for x in state) if isinstance(state,tuple) else edit(state)

def vector(state):
    if state is None:
        return []
    if isinstance(state, tuple):
        state = torch.cat([x.flatten() for x in state])
    return state.detach().cpu().flatten().tolist()
