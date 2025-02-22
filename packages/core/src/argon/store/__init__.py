import argon.graph
import argon.typing as atyp
import argon.numpy as npx
import importlib

from argon.graph import GraphDef, GraphLeaves, Node

from flax.nnx.graph import NodeDef, NodeRef, VariableDef, HashableMapping

import typing as typ
import jaxlib.xla_extension
import numpy as np
import zarr
import jaxlib
from zarr.storage import StorePath, StoreLike

def _type_to_identifier(tp: typ.Type) -> str:
    if tp is None or tp == type(None):
        return "none"
    elif tp == atyp.Array or tp == jaxlib.xla_extension.ArrayImpl: return "array"
    elif tp == str: return "str"
    elif tp == int: return "int"
    elif tp == float: return "float"
    elif tp == bool: return "bool"
    elif tp == list: return "list"
    elif tp == dict: return "dict"
    elif tp == tuple: return "tuple"
    elif tp == set: return "set"
    elif tp == frozenset: return "frozenset"
    else:
        return f"{tp.__module__}.{tp.__name__}"

def _type_from_identifier(identifier: str) -> typ.Type:
    if identifier == "none": return type(None)
    elif identifier == "array": return atyp.Array
    elif identifier == "str": return str
    elif identifier == "int": return int
    elif identifier == "float": return float
    elif identifier == "bool": return bool
    elif identifier == "list": return list
    elif identifier == "dict": return dict
    elif identifier == "tuple": return tuple
    elif identifier == "set": return set
    elif identifier == "frozenset": return frozenset
    else:
        idx = identifier.rfind(".")
        module, name = identifier[:idx], identifier[idx+1:]
        module = importlib.import_module(module)
        return getattr(module, name)

def _pack_graphdef(graphdef: GraphDef) -> dict:
    if isinstance(graphdef, NodeDef):
        return {
            "graph_type": "node",
            "type": _type_to_identifier(graphdef.type),
            "index": graphdef.index,
            "attributes": graphdef.attributes,
            "metadata": graphdef.metadata,
            "static_fields": dict(graphdef.static_fields),
            "index_mapping": dict(graphdef.index_mapping) if graphdef.index_mapping is not None else None,
            "subgraphs": {
                k: _pack_graphdef(v) for (k, v) in graphdef.subgraphs.items()
            },
            "leaves": {
                k: _pack_graphdef(v) for (k, v) in graphdef.leaves.items()
            },
        }
    elif isinstance(graphdef, NodeRef):
        return {
            "graph_type": "ref",
            "type": _type_from_identifier(graphdef.type),
            "index": graphdef.index
        }
    elif isinstance(graphdef, VariableDef):
        return {
            "graph_type": "variable",
            "type": _type_to_identifier(graphdef.type),
            "index": graphdef.index,
            "metadata": dict(graphdef.metadata),
        }


def _unpack_graphdef(packed: dict) -> GraphDef:
    graph_type = packed["graph_type"]
    if graph_type == "node":
        return NodeDef(
            type=_type_from_identifier(packed["type"]),
            index=packed["index"],
            attributes=tuple(packed["attributes"]),
            metadata=packed["metadata"],
            static_fields=HashableMapping({
                int(k) if isinstance(k, str) and k.isdigit() else k: v 
                for (k,v) in packed["static_fields"].items()
            }),
            index_mapping=HashableMapping(packed["index_mapping"]) if packed["index_mapping"] is not None else None,
            subgraphs=HashableMapping({
                int(k) if isinstance(k, str) and k.isdigit() else k: _unpack_graphdef(v) 
                for (k, v) in packed["subgraphs"].items()
            }),
            leaves=HashableMapping({
                int(k) if isinstance(k, str) and k.isdigit() else k: _unpack_graphdef(v)
                for (k, v) in packed["leaves"].items()
            }),
        )
    elif graph_type == "ref":
        return NodeRef(
            type=_type_from_identifier(packed["type"]),
            index=packed["index"]
        )
    elif graph_type == "variable":
        return VariableDef(
            type=_type_from_identifier(packed["type"]),
            index=packed["index"],
            metadata=HashableMapping(packed["metadata"]),
        )

def _flatten_state(state: GraphLeaves, prefix : str = ""):
    for k, v in state.items():
        if isinstance(v, typ.Mapping):
            yield from _flatten_state(v, prefix=f"{prefix}{k}/")
        else:
            yield f"{prefix}{k}", v

def load(store : StoreLike, *, 
         lazy : bool = False,
         path : StorePath | None = None,
         storage_options=None) -> Node:
    group = zarr.open_group(
        store, mode="r", 
        path=path,
        storage_options=storage_options
    )
    graphdef = _unpack_graphdef(group.attrs["graphdef"])

    def _unflatten_state(group):
        leaves = {}
        for key, value in group.members():
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            if isinstance(value, zarr.Group):
                leaves[key] = _unflatten_state(value)
            elif isinstance(value, zarr.Array):
                leaves[key] = value if lazy else npx.array(value)
        return leaves
    state = GraphLeaves(_unflatten_state(group), _copy=False)
    return argon.graph.merge(graphdef, state)

def dump(data : Node, store : StoreLike, *, path : StorePath | None = None):
    graphdef, state = argon.graph.split(data)
    group = zarr.create_group(store, path=path)
    graphdef = _pack_graphdef(graphdef)
    group.attrs["graphdef"] = graphdef
    for k, v in _flatten_state(state):
        group[k] = np.array(v)