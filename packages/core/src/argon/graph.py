import typing as tp
import numpy as np
from flax.nnx.graph import (
    PathParts, RefMap, Index, StateLeaf, NodeDef, NodeRef, get_node_impl,
    is_node, is_node_type, get_node_impl_for_type, GraphNodeImpl, Key, 
    VariableDef, Variable, HashableMapping,
    Node, KeyT, NodeLeaf, SubGraphAttribute, StaticAttribute, LeafAttribute
)
import argon.typing as atp
import jax.tree
import jax.tree_util
import typing

from flax.nnx.graph import (
    GraphDef, GraphState as GraphLeaves,
    GraphNodeImpl,
    update, update_context,
    split, split_context, 
    merge, merge_context,
    state as leaves,
    register_graph_node_type as _register_graph_node_type,
    register_pytree_node_type as _register_pytree_node_type,
    is_graph_node, Leaf, AuxData, Node
)
from flax.nnx import (
    display
)

class RawArray(Variable):
   pass

def _wrap_raw(value):
    return RawArray(value) if isinstance(value, (np.ndarray, atp.Array)) else value
def _wrap_raws(values):
    return tuple(_wrap_raw(x) for x in values)

def _unwrap_leaf(x):
    return x.value if isinstance(x, RawArray) else x
def _unwrap_leaves(x):
    return tuple(_unwrap_leaf(x) for x in x)


def register_graph_node_type(
        type: type,
        flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
        set_key: tp.Callable[[Node, Key, Leaf], None],
        pop_key: tp.Callable[[Node, Key], Leaf],
        create_empty: tp.Callable[[AuxData], Node],
        clear: tp.Callable[[Node], None],
        init: tp.Callable[[Node, tp.Iterable[tuple[Key, Leaf]]], None]):
    def _flatten(node):
        children, aux = flatten(node)
        children = [(k, _wrap_raw(v)) for k, v in children]
        return children, aux
    def _pop_key(node, key):
        value = pop_key(node, key)
        value = _wrap_raw(value)
        return value
    def _init(node, keys):
        new_keys = [(k, _unwrap_leaf(v)) for k, v in keys]
        return init(node, new_keys)
    def _set_key(node, key, value):
        value = _unwrap_leaf(value)
        if isinstance(value, RawArray):
            value = value.value
        return set_key(node, key, value)
    _register_graph_node_type(type,
        _flatten,
        _set_key,
        _pop_key,
        create_empty,
        clear,
        _init
    )

def register_pytree_node_type(type, flatten, unflatten, *, jax_register=True):
    def _nnx_flatten(node):
        children, aux = flatten(node)
        children = [(k, _wrap_raw(v)) for k, v in children]
        return children, aux
    def _nnx_unflatten(children, aux):
        children = [(k, _unwrap_leaf(v)) for k, v in children]
        return unflatten(children, aux)
    _register_pytree_node_type(type, _nnx_flatten, _nnx_unflatten)

    def _jax_flatten(node):
      children, aux = flatten(node)
      k = tuple(k for k, _ in children)
      v = tuple(v for _, v in children)
      return v, (k, aux)
    def _jax_unflatten(aux, children):
      k, aux = aux
      children = tuple(zip(k,children))
      return unflatten(children, aux)
    if jax_register:
        jax.tree_util.register_pytree_node(type, _jax_flatten, _jax_unflatten)

def map(f, graph : Node, *graphs : typing.Sequence[Node]) -> Node:
    graphdef, leaves = split(graph)
    other_leaves = [split(g)[1] for g in graphs]
    leaves = jax.tree.map(f, leaves, *other_leaves)
    return merge(graphdef, leaves)

# Re-register the basic pytree node types
# to allow for the raw array wrapper
from flax.nnx.graph import PYTREE_REGISTRY, PYTREE_NODE_IMPL, _key_path_to_key
def _flatten_pytree(pytree: tp.Any):
  leaves, treedef = jax.tree_util.tree_flatten_with_path(
    pytree, is_leaf=lambda x: x is not pytree
  )
  nodes = tuple((_key_path_to_key(path[0]), _wrap_raw(value)) for path, value in leaves)
  return nodes, treedef

def _unflatten_pytree(
  nodes: tuple[tuple[Key, tp.Any], ...], treedef: jax.tree_util.PyTreeDef
):
  pytree = treedef.unflatten(_unwrap_leaf(value) for _, value in nodes)
  return pytree

object.__setattr__(PYTREE_NODE_IMPL, "flatten", _flatten_pytree)
object.__setattr__(PYTREE_NODE_IMPL, "unflatten", _unflatten_pytree)

del PYTREE_REGISTRY[list]
del PYTREE_REGISTRY[tuple]
del PYTREE_REGISTRY[dict]
del PYTREE_REGISTRY[type(None)]

register_pytree_node_type(
  list,
  flatten=lambda x: (list(enumerate(x)), None),
  unflatten=lambda nodes, _: [value for _, value in nodes],  # type: ignore
  jax_register=False
)
# tuple
register_pytree_node_type(
  tuple,
  flatten=lambda x: (list(enumerate(x)), None),
  unflatten=lambda nodes, _: tuple(value for _, value in nodes),  # type: ignore
  jax_register=False
)
# dict
register_pytree_node_type(
  dict,
  flatten=lambda x: (sorted(x.items()), None),
  unflatten=lambda nodes, _: {key: value for key, value in nodes},  # type: ignore
  jax_register=False
)
# None
register_pytree_node_type(
  type(None),
  flatten=lambda x: ([], None),
  unflatten=lambda _, __: None,  # type: ignore
  jax_register=False
)

# register the GraphState, VariableState types as a pytree node
# so that we can serialize it easily
# Register the GraphState type as a pytree node
from flax.nnx.statelib import State as _State
from flax.nnx.graph import VariableState
def _state_flatten(x: _State):
  items = sorted(x._mapping.items())
  return items, None

def _state_unflatten(children, aux):
  return _State(children)

register_pytree_node_type(
  _State,
  _state_flatten,
  _state_unflatten,
  jax_register=False
)

def _variable_state_flatten(x: VariableState[tp.Any]):
  metadata = tuple(x.get_metadata().items())
  static = (x.type, metadata) if len(metadata) > 0 else x.type
  return (("value", x.value),), static

def _variable_state_unflatten(
    children: tuple[tp.Any],
    static: tuple[tp.Type, tuple[tuple[str, tp.Any], ...]],
) -> VariableState:
  if isinstance(static, typing.Type):
     static = (static, ())
  return VariableState(
    type=static[0],
    value=children[0][1],
    **dict(static[1]),
  )

register_pytree_node_type(
   VariableState, 
    _variable_state_flatten,
    _variable_state_unflatten,
   jax_register=False
)