from dataclasses import dataclass

@dataclass(frozen=True)
class Rewards:
    escape: float = 10.0
    key: float = 1.0
    unlock: float = 0.5
    discovery: float = 0.1
    step: float = -0.01
    invalid: float = -0.05
    loop: float = -0.02
    wrong_seal: float = -2.0
