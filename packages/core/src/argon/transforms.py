# Monkey-patch first!
import argon.graph
from argon.struct import struct

from flax.nnx import (
    grad, value_and_grad,
    scan, vmap,
    StateSharding,
    eval_shape, switch, cond,
    while_loop, fori_loop,
    Carry
)
from jax import pure_callback, devices

import typing as tp
import functools
import dataclasses
import jax

import argon.numpy as npx

from flax.typing import Missing
from flax.nnx import extract, graph
from flax.nnx.transforms.compilation import JitFn

F = tp.TypeVar('F', bound=tp.Callable)

def _jit_split_fn(ctx: graph.SplitContext, path, prefix, x):
  if graph.is_graph_node(x):
    if isinstance(prefix, StateSharding):
        return extract.NodeStates.from_split(
        *ctx.split(x, *prefix.filters), metadata=prefix
        )
    return extract.NodeStates.from_split(*ctx.split(x))
  else:
    if isinstance(x, (jax.Array)):
        return x
    else:
        return graph.Static(x)

@dataclasses.dataclass(eq=False)
class JitFn:
  f: tp.Callable[..., tp.Any]
  in_shardings: tp.Any
  out_shardings: tp.Any
  kwarg_shardings: tp.Any

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args, **pure_kwargs):
    pure_args, pure_kwargs = jax.tree.map(
      lambda v: v.value if isinstance(v, graph.Static) else v,
      (pure_args, pure_kwargs),
      is_leaf=lambda v: isinstance(v, graph.Static),
    )
    args, kwargs = extract.from_tree((pure_args, pure_kwargs), ctxtag='jit')

    out = self.f(*args, **kwargs)

    args_out, kwargs_out = extract.clear_non_graph_nodes((args, kwargs))
    pure_args_out, pure_kwargs_out, pure_out = extract.to_tree(
      (args_out, kwargs_out, out),
      prefix=(self.in_shardings, self.kwarg_shardings, self.out_shardings),
      ctxtag='jit',
      split_fn=_jit_split_fn
    )

    return pure_args_out, pure_kwargs_out, pure_out

def jit(
  fun: F | type[Missing] = Missing,
  *,
  in_shardings: tp.Any = None,
  out_shardings: tp.Any = None,
  static_argnums: int | tp.Sequence[int] | None = None,
  static_argnames: str | tp.Iterable[str] | None = None,
  donate_argnums: int | tp.Sequence[int] | None = None,
  donate_argnames: str | tp.Iterable[str] | None = None,
  inline: bool = False,
  abstracted_axes: tp.Optional[tp.Any] = None,
) -> F | tp.Callable[[F], F]:
  if fun is Missing:
    return functools.partial(
      jit,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      donate_argnames=donate_argnames,
      inline=inline,
      abstracted_axes=abstracted_axes,
    )  # type: ignore[return-value]
  kwarg_shardings = None
  jax_in_shardings = jax.tree.map(
    lambda x: extract.NodeStates.from_prefixes(x.shardings, metadata=x)
    if isinstance(x, StateSharding)
    else x,
    in_shardings,
  )
  jax_out_shardings = jax.tree.map(
    lambda x: extract.NodeStates.from_prefixes(x.shardings, metadata=x)
    if isinstance(x, StateSharding)
    else x,
    out_shardings,
  )

  jitted_fn = jax.jit(
    JitFn(fun, in_shardings, out_shardings, kwarg_shardings),
    in_shardings=jax_in_shardings,
    out_shardings=(jax_in_shardings, kwarg_shardings, jax_out_shardings),  # type: ignore
    static_argnums=static_argnums,
    static_argnames=static_argnames,
    donate_argnums=donate_argnums,
    donate_argnames=donate_argnames,
    inline=inline,
    abstracted_axes=abstracted_axes,
  )

  @functools.wraps(fun)
  @graph.update_context('jit')
  def jit_wrapper(*args, **kwargs):
    pure_args, pure_kwargs = extract.to_tree(
      (args, kwargs),
      prefix=(in_shardings, kwarg_shardings),
      split_fn=_jit_split_fn,
      check_aliasing=in_shardings is not None,
      map_non_graph_nodes=True,
      ctxtag='jit',
    )
    pure_args_out, pure_kwargs_out, pure_out = jitted_fn(
      *pure_args, **pure_kwargs
    )
    _args_out, _kwargs_out, out = extract.from_tree(
      (pure_args_out, pure_kwargs_out, pure_out), ctxtag='jit'
    )
    return out
  jit_wrapper.__str__ = fun.__str__
  jit_wrapper.__repr__ = fun.__repr__
  jit_wrapper.inner = jitted_fn  # type: ignore
  return jit_wrapper  # type: ignore


@struct(frozen=True)
class Partial:
  func: tp.Any
  args: tp.Tuple | None
  kwargs: tp.Dict[str, tp.Any] | None

  def __call__(self, *args, **kwargs):
    mod_kwargs = dict(self.kwargs)
    mod_kwargs.update(kwargs)
    return self.func(*self.args, *args, **mod_kwargs)

def partial(func: F, *args, **kwargs) -> F:
  return Partial(func, args, kwargs)

def map(func: F, batch_size: int, in_axes: tp.Any = 0, out_axes: tp.Any = 0) -> F:
  vmapped = vmap(func, in_axes=in_axes, out_axes=out_axes)
  @scan(in_axes=(Carry, in_axes))
  def scan_fn(carry, inputs):
    return carry, vmapped(*inputs)
  @jit
  def transformed(*args):
    leaves, _ = argon.tree.flatten(argon.tree.map(
      lambda axis, x: argon.tree.axis_size(x, axis) if axis is not None else None, 
      in_axes, args, is_leaf=lambda x: x is None
    ))
    leaves = [x for x in leaves if x is not None]
    N = leaves[0]
    batches = N // batch_size
    remainder = N - batches * batch_size
    def reshape_inputs(axis, x):
      if axis is None: return x
      else:
        if remainder == 0:
          return argon.tree.map(
            lambda x: npx.reshape(npx.moveaxis(x, axis, 0),
                  (batches, batch_size) + x.shape[1:]), x)
        else:
          return argon.tree.map(
            lambda x: npx.reshape(npx.moveaxis(x, axis, 0)[:-remainder],
                  (batches, batch_size) + x.shape[1:]), x)
    def reshape_remainder(axis, x):
      if axis is None: return x
      else:
        return argon.tree.map(
          lambda x: npx.moveaxis(x, axis, 0)[-remainder:], x)
    bulk = argon.tree.map(reshape_inputs, in_axes, args, is_leaf=lambda x: x is None) if batches > 0 else None
    rest = argon.tree.map(reshape_remainder, in_axes, args, is_leaf=lambda x: x is None) if remainder > 0 else None
    _, bulk_out = scan_fn(None, bulk) if bulk is not None else (None, None)
    rest_out = vmapped(*rest) if rest is not None else None
    if bulk_out is None:
      return rest_out
    elif rest_out is None:
      return argon.tree.map(
        lambda x: npx.reshape(x, (N,) + x.shape[2:]), bulk_out
      )
    else:
      combined = argon.tree.map(
        lambda x, y: npx.concatenate([
          x.reshape(x.shape[0]*x.shape[1], *x.shape[2:]), y
        ]), bulk_out, rest_out)
      return combined
  return transformed