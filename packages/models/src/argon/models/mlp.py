import argon.nn as nn
import argon.typing as atyp
from argon.registry import Registry
from argon.nn import activation as activations

import typing as typ
from functools import partial

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                    hidden_features: typ.Sequence[int],
                    activation: callable = activations.relu, *, rngs: nn.Rngs):
        features = (in_features,) + tuple(hidden_features) + (out_features,)
        self.layers = tuple(
            nn.Linear(i, o, rngs=rngs) for i, o in zip(features[:-1], features[1:])
        )
        self.activation = activation

    def __call__(self, x: atyp.Array):
        assert x.ndim == 1
        h = self.activation
        for layer in self.layers[:-1]:
            x = h(layer(x))
        return self.layers[-1](x)

def register(registry: Registry[nn.Module], prefix=None):
    registry.register("mlp", MLP)