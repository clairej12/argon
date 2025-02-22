import argon.numpy as npx
import argon.nn as nn
import argon.typing as atyp
import typing as typ
import functools
from flax.nnx.helpers import has_keyword_arg

from typing import Sequence, Callable

ConvDef = Callable[[], nn.Conv]
NormDef = Callable[[], nn.Module]
ActivationDef = Callable[[atyp.Array], atyp.Array]
Embed = Callable[[typ.Any], atyp.Array]

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int | Sequence[int], 
                strides: int | Sequence[int] = 1,
                *,
                operation: Callable[[atyp.Array], atyp.Array] | None = None,
                dropout: float | None = None,
                skip_full_conv: bool = False,

                cond_features: int | None = None,
                use_film: bool = True,

                conv: Callable[[], nn.Module] = nn.Conv,
                norm: Callable[[], nn.Module] = nn.BatchNorm,
                activation: Callable[[atyp.Array], atyp.Array] = nn.activations.relu,
                rngs: nn.Rngs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(strides, int):
            strides = (strides,) * len(kernel_size)

        self.activation = activation
        self.operation = operation

        if dropout:
            self.dropout = nn.Dropout(dropout, rngs=rngs)
        if cond_features is not None:
            self.embed_layers = nn.Sequential(
                nn.Linear(cond_features, cond_features, rngs=rngs),
                activation,
                nn.Linear(cond_features, 2*out_channels if use_film else out_channels, rngs=rngs),
            )

        self.norm_a = norm(in_channels, rngs=rngs)
        self.norm_b = norm(out_channels, rngs=rngs)

        self.conv_a = conv(
            in_channels, out_channels,
            kernel_size, strides, rngs=rngs,
        )
        self.conv_b = conv(
            out_channels, out_channels,
            kernel_size, strides, rngs=rngs
        )
        if in_channels != out_channels or skip_full_conv:
            self.conv_proj = conv(
                in_channels, out_channels,
                (3,)*len(kernel_size) if skip_full_conv else (1,)*len(kernel_size), 
                strides if skip_full_conv else (1,)*len(kernel_size), rngs=rngs
            )

    def __call__(self, x, cond=None, target_shape=None):
        residual = x

        h = self.norm_a(x)
        h = self.activation(h)

        if self.operation is not None:
            if target_shape is not None and has_keyword_arg(self.operation, 'target_shape'):
                op = functools.partial(self.operation, target_shape=target_shape)
            else:
               op = self.operation
            residual = op(residual)
            h = op(h)

        h = self.conv_a(h)

        if cond is not None:
            embed = self.embed_layers(cond)
            if embed.shape[-1] == 2*h.shape[-1]:
                shift, scale = npx.split(embed, 2, axis=-1)
                h = self.norm_b(h) * (1 + scale) + shift
            else:
                h = self.norm_b(h + embed)
        else:
            h = self.norm_b(h)

        h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.conv_b(h)
        # If the input and output shapes are different, 
        # we need to project the input
        if h.shape[-1] != residual.shape[-1]:
            residual = self.conv_proj(residual)
        return h + residual

class CondSequential(nn.Module):
  def __init__(self, *fns: typ.Callable[..., typ.Any]):
    self.layers = list(fns)

  def __call__(self, *args, 
                rngs: nn.Rngs | None = None, 
                cond: atyp.Array | None = None, 
                target_shape: typ.Sequence[int] | None = None,
                **kwargs) -> typ.Any:
    for i, f in enumerate(self.layers):
      if not callable(f):
        raise TypeError(f'Sequence[{i}] is not callable: {f}')
      if i > 0:
        if isinstance(output, tuple):
          args = output
          kwargs = {}
        elif isinstance(output, dict):
          args = ()
          kwargs = output
        else:
          args = (output,)
          kwargs = {}
      if rngs is not None and has_keyword_arg(f, 'rngs'):
        kwargs['rngs'] = rngs
      if cond is not None and has_keyword_arg(f, 'cond'):
        kwargs['cond'] = cond
      if target_shape is not None and has_keyword_arg(f, 'target_shape'):
        kwargs['target_shape'] = target_shape
      output = f(*args, **kwargs)

    return output