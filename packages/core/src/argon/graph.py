import typing as tp
import numpy as np
from flax.nnx.graph import (
    PathParts, RefMap, Index, StateLeaf, NodeDef, NodeRef, get_node_impl,
    is_node, GraphNodeImpl, Key, VariableDef, Variable, HashableMapping,
    Node, KeyT, NodeLeaf
)
import flax.nnx.graph

# A "variable" type which is just a raw array
class RawArray(Variable):
  def from_metadata(value, metadata):
    return value

def _graph_flatten(
  path: PathParts,
  ref_index: RefMap[tp.Any, Index],
  flat_state: dict[PathParts, StateLeaf],
  node: Node,
) -> NodeDef[Node] | NodeRef:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if node in ref_index:
    return NodeRef(type(node), ref_index[node])

  node_impl = get_node_impl(node)

  # only cache graph nodes
  if isinstance(node_impl, GraphNodeImpl):
    index = len(ref_index)
    ref_index[node] = index
  else:
    index = -1

  subgraphs: list[tuple[Key, NodeDef[Node] | NodeRef]] = []
  static_fields: list[tuple[Key, tp.Any]] = []
  leaves: list[tuple[Key, VariableDef | NodeRef]] = []

  values, metadata = node_impl.flatten(node)
  for key, value in values:
    if is_node(value):
      nodedef = _graph_flatten((*path, key), ref_index, flat_state, value)
      subgraphs.append((key, nodedef))
    elif isinstance(value, Variable):
      if value in ref_index:
        leaves.append((key, NodeRef(type(value), ref_index[value])))
      else:
        flat_state[(*path, key)] = value.to_state()
        variable_index = ref_index[value] = len(ref_index)
        variabledef = VariableDef(
          type(value), variable_index, HashableMapping(value.get_metadata())
        )
        leaves.append((key, variabledef))
    elif isinstance(value, (jax.Array, np.ndarray)):
      if value in ref_index:
        leaves.append((key, NodeRef(type(value), ref_index[value])))
      else:
        flat_state[(*path, key)] = value
        variable_index = ref_index[value] = len(ref_index)
        variabledef = VariableDef(
          RawArray, variable_index, HashableMapping({})
        )
        leaves.append((key, variabledef))
    else:
      static_fields.append((key, value))

  nodedef = NodeDef.create(
    type=node_impl.type,
    index=index,
    attributes=tuple(key for key, _ in values),
    subgraphs=subgraphs,
    static_fields=static_fields,
    leaves=leaves,
    metadata=metadata,
    index_mapping=None,
  )
  return nodedef

# Monkeypatch to support arrays in leaves
flax.nnx.graph._graph_flatten = _graph_flatten

def _graph_unflatten(
  nodedef: NodeDef[Node] | NodeRef[Node],
  state: tp.Mapping[KeyT, StateLeaf | tp.Mapping[Key, tp.Any]],
  index_ref: dict[Index, tp.Any],
  index_ref_cache: dict[Index, tp.Any] | None,
) -> Node:
  """Recursive helper for graph_unflatten.

  Args:
    nodedef: A GraphDef instance or an index to a node in the cache.
    state: A mapping from attribute names to variables or subgraphs.
    index_to_ref: A mapping from indexes to nodes that have been traversed.
      If a node is already in the cache, it won't be traversed again.
    index_ref_cache: A mapping from indexes to existing nodes that can be reused.
      When an reference is reused, ``GraphNodeImpl.clear`` is called to leave the
      object in an empty state and then filled by the unflatten process, as a result
      existing graph nodes are mutated to have the new content/topology
      specified by the nodedef.
  """
  if isinstance(nodedef, NodeRef):
    return index_ref[nodedef.index]

  if not flax.nnx.graph.is_node_type(nodedef.type):
    raise RuntimeError(f'Unsupported type: {nodedef.type}, this is a bug.')

  if nodedef.index in index_ref:
    raise RuntimeError(f'GraphDef index {nodedef.index} already used.')

  node_impl = flax.nnx.graph.get_node_impl_for_type(nodedef.type)

  def _get_children():
    children: dict[Key, NodeLeaf | Node] = {}

    # NOTE: we could allw adding new StateLeafs here
    if unkown_keys := set(state) - set(nodedef.attributes):
      raise ValueError(f'Unknown keys: {unkown_keys}')

    # for every key in attributes there are 6 possible cases:
    #  - (2) the key can either be present in the state or not
    #  - (3) the key can be a subgraph, a leaf, or a static attribute
    for key in nodedef.attributes:
      if key not in state:
        # if key is not present create an empty types
        if key in nodedef.static_fields:
          children[key] = nodedef.static_fields[key]
        elif key in nodedef.subgraphs:
          # if the key is a subgraph we create an empty node
          subgraphdef = nodedef.subgraphs[key]
          assert not isinstance(subgraphdef, VariableDef)
          if isinstance(subgraphdef, NodeRef):
            # subgraph exists, take it from the cache
            children[key] = index_ref[subgraphdef.index]
          else:
            # create a node from an empty state, reasoning:
            # * its a node with no state
            # * its a node with state but only through references of already
            #   created nodes
            substate = {}
            children[key] = _graph_unflatten(
              subgraphdef, substate, index_ref, index_ref_cache
            )
        elif key in nodedef.leaves:
          variabledef = nodedef.leaves[key]
          if variabledef.index in index_ref:
            # variable exists, take it from the cache
            children[key] = index_ref[variabledef.index]
          else:
            # key for a variable is missing, raise an error
            raise ValueError(
              f'Expected key {key!r} in state while building node of type '
              f'{nodedef.type.__name__}.'
            )
        else:
          raise RuntimeError(f'Unknown static field: {key!r}')
      else:
        value = state[key]
        if key in nodedef.static_fields:
          raise ValueError(
            f'Got state for static field {key!r}, this is not supported.'
          )
        if key in nodedef.subgraphs:
          if flax.nnx.graph.is_state_leaf(value):
            raise ValueError(
              f'Expected value of type {nodedef.subgraphs[key]} for '
              f'{key!r}, but got {value!r}'
            )
          assert isinstance(value, dict)
          subgraphdef = nodedef.subgraphs[key]

          if isinstance(subgraphdef, NodeRef):
            children[key] = index_ref[subgraphdef.index]
          else:
            children[key] = _graph_unflatten(
              subgraphdef, value, index_ref, index_ref_cache
            )

        elif key in nodedef.leaves:
          variabledef = nodedef.leaves[key]

          if variabledef.index in index_ref:
            # add an existing variable
            assert isinstance(variabledef, NodeRef)
            children[key] = index_ref[variabledef.index]
          else:
            # its a unseen variable, create a new one
            assert isinstance(variabledef, VariableDef)
            # when idxmap is present, check if the Varable exists there
            # and update existing variables if it does
            if (
              index_ref_cache is not None
              and variabledef.index in index_ref_cache
            ):
              # if variable exists, update it
              variable = index_ref_cache[variabledef.index]
              if isinstance(variable, (jax.Array, np.ndarray)):
                # override the value if it is an array
                children[key] = variable
                index_ref[variabledef.index] = variable
              elif not isinstance(variable, Variable):
                raise ValueError(
                  f'Expected a Variable type for {key!r}, but got {type(variable)}.'
                )
              if isinstance(value, flax.nnx.VariableState):
                variable.update_from_state(value)
              else:
                variable.raw_value = value
            else:  # if it doesn't, create a new variable
              if isinstance(value, flax.nnx.VariableState):
                variable = value.to_variable()
              else:
                variable = variabledef.type.from_metadata(
                  value, variabledef.metadata
                )
            children[key] = variable
            index_ref[variabledef.index] = variable
        else:
          raise RuntimeError(f'Unknown key: {key!r}, this is a bug.')

    return children

  if isinstance(node_impl, GraphNodeImpl):
    # we create an empty node first and add it to the index
    # this avoids infinite recursion when there is a reference cycle
    if index_ref_cache is not None and nodedef.index in index_ref_cache:
      node = index_ref_cache[nodedef.index]
      if type(node) != nodedef.type:
        raise ValueError(
          f'Expected a node of type {nodedef.type} for index '
          f'{nodedef.index}, but got a node of type {type(node)}.'
        )
      node_impl.clear(node)
    else:
      node = node_impl.create_empty(nodedef.metadata)
    index_ref[nodedef.index] = node
    children = _get_children()
    node_impl.init(node, tuple(children.items()))
  else:
    # if the node type does not support the creation of an empty object it means
    # that it cannot reference itself, so we can create its children first
    children = _get_children()
    node = node_impl.unflatten(tuple(children.items()), nodedef.metadata)

  return node

flax.nnx.graph._graph_unflatten = _graph_unflatten

import jax.tree
import jax.tree_util
import typing

from flax.nnx.graph import (
    GraphDef, GraphState as GraphLeaves,
    update, update_context,
    split, split_context, 
    merge, merge_context,
    state as leaves,
    register_graph_node_type,
    register_pytree_node_type as _register_pytree_node_type,
    is_graph_node
)
from flax.nnx import (
    display
)


def register_pytree_node_type(type, flatten, unflatten):
    _register_pytree_node_type(type, flatten, unflatten)
    def _jax_flatten(node):
      children, aux = flatten(node)
      k = tuple(k for k, _ in children)
      v = tuple(v for _, v in children)
      return v, (k, aux)
    def _jax_unflatten(aux, children):
      k, aux = aux
      children = tuple(zip(k,children))
      return unflatten(children, aux)
    jax.tree_util.register_pytree_node(type, _jax_flatten, _jax_unflatten)

def map(f, graph : Node, *graphs : typing.Sequence[Node]) -> Node:
    graphdef, leaves = split(graph)
    other_leaves = [split(g)[1] for g in graphs]
    leaves = jax.tree.map(f, leaves, *other_leaves)
    return merge(graphdef, leaves)