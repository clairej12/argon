import argon.typing as atyp

from flax.nnx import (
    Module, 
    Linear, Conv, ConvTranspose,
    Sequential,
    Dropout, BatchNorm, GroupNorm, LayerNorm,
    Rngs, Optimizer, Variable,
    MultiHeadAttention
)
from . import activation as activations
import flax.linen
import typing as typ
import jax

def upsample(x: atyp.Array, *,
        scale_factors: typ.Sequence[int] | None = None,
        target_shape: typ.Sequence[int] | None = None):
    assert scale_factors is not None or target_shape is not None
    if scale_factors is not None:
        assert len(scale_factors) == x.ndim - 1
        dest_shape = tuple(
            s * f for s, f in zip(x.shape[1:], scale_factors)
        ) + x.shape[-1:]
    else:
        assert len(target_shape) == x.ndim - 1
        dest_shape = tuple(target_shape) + x.shape[-1:]
    return jax.image.resize(x, dest_shape, method="nearest")

def downsample(x: atyp.Array, scale_factors: typ.Sequence[int] | None = None):
    if scale_factors is None:
        scale_factors = (2,) * (x.ndim - 1)
    return flax.linen.avg_pool(
        x, window_shape=scale_factors, strides=scale_factors
    )

class Identity(Module):
    def __call__(self, x: atyp.Array) -> atyp.Array:
        return x