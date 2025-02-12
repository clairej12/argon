from argon.core import Array
from argon.core.dataclasses import dataclass
from argon.policy import Policy, PolicyInput, PolicyOutput

class LinearPolicy(Policy):
    K: Array

    def __call__(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        return PolicyOutput(K @ obs)