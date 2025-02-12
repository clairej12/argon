from __future__ import annotations

import dataclasses as dcls
import functools
import weakref
import jax.tree_util
import jax.flatten_util

import typing as ty
from types import NoneType
from typing import Callable
from collections.abc import Iterable, Sequence
from collections import namedtuple

from numpy._core.numeric import identity

Node = ty.TypeVar("Node")
AuxData = ty.TypeVar("AuxData")
AuxDiff = ty.TypeVar("AuxDiff")
Leaf = ty.TypeVar("Leaf")
K = ty.TypeVar("K")
V = ty.TypeVar("V")
Id = int

class Key(ty.Hashable, ty.Protocol):
  def __lt__(self: K, value: K, /) -> bool: ...

KeyPath = Sequence[Key]

def is_key_like(x: ty.Any) -> ty.TypeGuard[Key]:
  return hasattr(x, '__hash__') and hasattr(x, '__lt__')

# The underlying operations a Graph Node needs to implement
@dcls.dataclass(frozen=True, slots=True)
class NodeImpl(ty.Generic[Node, AuxData, AuxDiff]):
    type: ty.Type[Node]
    identifier: str
    version: str
    # For regular pytree types
    unflatten: Callable[[AuxData, Iterable[tuple[Key, Leaf]]], Node]
    flatten: Callable[[Node], tuple[ty.Sequence[tuple[Key, Leaf]], AuxData]]
    # For recursive-capable types (e.g. not tuples!)
    new: Callable[[], Node]
    init: Callable[[Node, AuxData, Iterable[tuple[Key, Leaf]]], None]
    # For mutable types
    aux_diff: Callable[[AuxData, AuxData], AuxDiff] | None
    replace: Callable[[Node, AuxDiff, Iterable[Key, Leaf]], None] | None

GRAPH_REGISTRY : weakref.WeakKeyDictionary[ty.Type, NodeImpl] = weakref.WeakKeyDictionary()
IDENTIFIER_REGISTRY : weakref.WeakValueDictionary[str, ty.Type]

def register_graph_type(type: Node, impl: NodeImpl[Node]):
    GRAPH_REGISTRY[type] = impl
    IDENTIFIER_REGISTRY[f"{impl.identifier}:{impl.version}"] = type

def lookup_registry_type(type: ty.Type[Node]) -> NodeImpl[Node] | None:
    return GRAPH_REGISTRY[type]

def lookup_registry_identifier(identifier: str, version: str) -> NodeImpl[Node] | None:
    type = IDENTIFIER_REGISTRY[f"{identifier}:{version}"]
    return register_graph_type(type)

# The graph definition code
class GraphDef(ty.Generic[Node]):
    type: ty.Type[Node]
    index: int

@dcls.dataclass(frozen=True, slots=True)
class LeafDef(ty.Generic[Node], GraphDef[Node]):
    type: ty.Type[Node]
    index: int

@dcls.dataclass(frozen=True, slots=True)
class NodeRef(ty.Generic[Node], GraphDef[Node]):
    type: ty.Type[Node]
    index: int

@dcls.dataclass(frozen=True, slots=True)
class NodeDef(ty.Generic[Node], GraphDef[Node]):
    type: ty.Type[Node]
    index: int
    aux_data: AuxData
    children: FrozenDict[Key, GraphDef] | None = None

jax.tree_util.register_static(NodeDef)
jax.tree_util.register_static(NodeRef)

class GraphLeaves:
    def __init__(self, leaves: ty.Mapping[Key, GraphLeaves | Leaf], copy: bool = True):
        self._mapping = dict(sorted(leaves.items())) if copy else leaves
    def __contains__(self, key: object) -> bool:
        return key in self._mapping
    def __getitem__(self, key: K) -> V:
        return self._mapping[key]
    def __repr__(self) -> str:
        return repr(self._mapping)

@dcls.dataclass(frozen=True, slots=True)
class GraphNodes:
    ref: Node
    children: FrozenDict[Key, GraphNodes]

# The flatten function!
def _flatten(graph: Node) -> tuple[GraphDef, GraphNodes, GraphLeaves]:
    node_indices : dict[Id, int] = {}
    def visit(node: Node) -> tuple[GraphDef, GraphNodes, GraphLeaves]:
        nid, nty = id(node), type(node)
        if nid in node_indices:
            return NodeRef(nty, node_indices[nid]), None, None
        index = len(node_indices)
        node_indices[nid] = index
        if nty not in GRAPH_REGISTRY:
            return LeafDef(nty, index), None, None

        impl = GRAPH_REGISTRY[nty]
        children, aux = impl.flatten(node)
        sub_leaves, sub_nodes = {}, {}
        for key, child in children:
            d, nodes, leaves = visit(child)
            if isinstance(d, LeafDef): sub_leaves[key] = d
            elif isinstance(d, NodeDef): sub_nodes[key] = d
        nodes = GraphNodes(
            ref=node,
            children=FrozenDict(sub_nodes, copy=False)
        )
        nodedef = NodeDef(nty, index, aux, children)
        return nodedef, nodes, leaves
    graphdef = visit(graph, [])

# Will populate refs in the graphdef
# with the actual values from a given value or the leaves

class NotFound: ...
NOT_FOUND = NotFound()

def _unflatten_populate_ref_map(
            ref_map: dict[int, Node],
            current_path: list[Key],
            graphdef: GraphDef[Node],
        ):
    pass

def _unflatten(
                leaves: GraphLeaves,
                # Provide a node value to unflatten into
                into_node: Node | None = None
            ) -> Node:
    if isinstance(graphdef, LeafDef):
        value = leaves[path]
        return value
    elif isinstance(graphdef, NodeRef):
        return refmap.get(graphdef.index, graphdef.type, leaves)
    elif isinstance(graphdef, NodeDef):
        children = []
        for key, child in graphdef.children.items():
            path.push(key)
            children.append((key, _unflatten(child, leaves, refmap, path)))
            path.pop()
        if graphdef.index is not None: 
            impl.init(value, graphdef.aux_data, children)
            refmap.mark_initialized(graphdef.index)
        else:
            value = impl.unflatten(graphdef.aux_data, children)
    return value

def graphdef(graph: Node) -> GraphDef:
    return _flatten(graph, False)

def flatten(graph: Node) -> tuple[GraphDef, GraphLeaves]:
    return _flatten(graph)

def unflatten(graphdef: GraphDef[Node], state: GraphLeaves) -> Node:
    pass

class FrozenDict(ty.Mapping[K, V], ty.Hashable):
  def __init__(self, mapping: ty.Mapping[K, V], copy: bool = True):
    self._mapping = dict(sorted(mapping.items())) if copy else mapping

  def __contains__(self, key: object) -> bool:
    return key in self._mapping

  def __getitem__(self, key: K) -> V:
    return self._mapping[key]

  def __iter__(self) -> ty.Iterator[V]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __hash__(self) -> int:
    return hash(tuple(sorted(self._mapping.items())))

  def __eq__(self, other: ty.Any) -> bool:
    return (isinstance(other, FrozenDict) and self._mapping == other._mapping)

  def __repr__(self) -> str:
    return repr(self._mapping)