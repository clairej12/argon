import argon.numpy as npx
import jax

from jax.tree import *

from jax.flatten_util import ravel_pytree

def shape(tree):
    return map(lambda x: npx.shape(x), tree)

def structure(tree):
    return map(lambda x: jax.ShapeDtypeStruct(npx.shape(x), x.dtype), tree)

from jax._src.api_util import flatten_axes as _flatten_axes
def axis_size(pytree, axes_tree = None, /) -> int:
    if axes_tree is None: axes_tree = 0
    args_flat, in_tree  = flatten(pytree)
    in_axes_flat = _flatten_axes("axis_size in_axes", in_tree, axes_tree, kws=False)
    axis_sizes_ = [x.shape[i] for x, i in zip(args_flat, in_axes_flat)]
    assert all(x == axis_sizes_[0] for x in axis_sizes_)
    return axis_sizes_[0]