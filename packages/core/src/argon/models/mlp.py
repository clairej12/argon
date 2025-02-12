import flax.linen as nn
import flax.linen.activation as activations

from argon.registry import Registry
from functools import partial


from collections.abc import Sequence

class MLP(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        assert x.shape == (self.input_dim,)
        h = getattr(activations, self.activation)
        for f in self.hidden_dim:
            x = nn.Dense(f)(x)
            x = h(x)
        x = nn.Dense(self.output_dim)(x)
        return x

def register(registry: Registry[nn.Module], prefix=None):
    pass
