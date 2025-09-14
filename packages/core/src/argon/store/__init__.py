import argon.graph
import argon.typing as atyp
import argon.numpy as npx
import importlib

from argon.graph import GraphDef, GraphLeaves, Node

from flax.nnx.graph import NodeDef, NodeRef, VariableDef, HashableMapping

import typing as typ
import types
import jaxlib.xla_extension
import numpy as np
import zarr
import jaxlib
from zarr.storage import StorePath, StoreLike, LocalStore

import pdb

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

def _pack_static(a):
    if a is None:
        return "__none__"
    if isinstance(a, typ.Type):
        return {"__type__": "type", "identifier": _type_to_identifier(a)}
    elif isinstance(a, np.ndarray):
        return {"__type__": "array", "shape": a.shape, "dtype": a.dtype}
    elif isinstance(a, atyp.ShapeDtypeStruct):
        return {"__type__": "shape_dtype_struct", "shape": a.shape, "dtype": a.dtype}
    elif isinstance(a, dict):
        if "__type__" in a:
            raise ValueError(f"Key '__type__' is reserved for static fields")
        a = { k: _pack_static(v) for (k, v) in a.items() }
        a["__type__"] = "dict"
    elif isinstance(a, tuple):
        return {
            "__type__" : "tuple", 
            "items" : [_pack_static(x) for x in a] 
        }
    elif isinstance(a, list):
        return {
            "__type__" : "list",
            "items" : [_pack_static(x) for x in a]
        }
    elif isinstance(a, set):
        return {
            "__type__" : "set",
            "items" : [_pack_static(x) for x in a]
        }
    elif isinstance(a, (int, float, bool, str)):
        if a == "__none__":
            raise ValueError(f"Value '__none__' is reserved for None")
        return a
    else:
        raise ValueError(f"Unsupported type {type(a)} for static fields")

def _unpack_static(a):
    if a == "__none__": return None
    if isinstance(a, dict):
        t = a["__type__"]
        match t:
            case "array": return np.empty(a["shape"], dtype=a["dtype"])
            case "shape_dtype_struct": return atyp.ShapeDtypeStruct(a["shape"], a["dtype"])
            case "type": return _type_from_identifier(a["identifier"])
            case "dict":
                a = dict(a)
                del a["__type__"]
                return { k: _unpack_static(v) for (k, v) in a.items() }
            case "tuple":
                return tuple(_unpack_static(x) for x in a["items"])
            case "list":
                return [_unpack_static(x) for x in a["items"]]
            case "set":
                return set(_unpack_static(x) for x in a["items"])
    else:
        return a

from flax.nnx.graph import SubGraphAttribute, StaticAttribute, LeafAttribute
def _pack_attribute(attr):
    if isinstance(attr, SubGraphAttribute):
        return {
            "type": "subgraph",
            "key": attr.key,
            "graphdef": _pack_graphdef(attr.value)
        }
    elif isinstance(attr, StaticAttribute):
        return {
            "type": "static",
            "key": attr.key,
            "value": _pack_static(attr.value)
        }
    elif isinstance(attr, LeafAttribute):
        return {
            "type": "leaf",
            "key": attr.key,
            "graphdef": _pack_graphdef(attr.value)
        }

def _unpack_attribute(attr):
    typ = attr["type"]
    match typ:
        case "subgraph": return SubGraphAttribute(attr["key"], _unpack_graphdef(attr["graphdef"]))
        case "static": return StaticAttribute(attr["key"], _unpack_static(attr["value"]))
        case "leaf": return LeafAttribute(attr["key"], _unpack_graphdef(attr["graphdef"]))

def _pack_graphdef(graphdef: GraphDef) -> dict:
    if isinstance(graphdef, NodeDef):
        return {
            "graph_type": "node",
            "type": _type_to_identifier(graphdef.type),
            "index": graphdef.index,
            "attributes": [_pack_attribute(a) for a in graphdef.attributes],
            "metadata": _pack_static(graphdef.metadata),
            "index_mapping": dict(graphdef.index_mapping) if graphdef.index_mapping is not None else None,
        }
    elif isinstance(graphdef, NodeRef):
        return {
            "graph_type": "ref",
            "type": _type_to_identifier(graphdef.type),
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
            metadata=_unpack_static(packed["metadata"]),
            index_mapping=HashableMapping(packed["index_mapping"]) 
                    if "index_mapping" in packed and packed["index_mapping"] is not None else None,
            attributes=tuple(
                _unpack_attribute(a) for a in packed["attributes"]
            ),
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
            metadata=HashableMapping({
                k : _unpack_static(v) for (k, v) in packed["metadata"].items()
            }),
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
                if "dtype" in value.attrs and value.attrs["dtype"] == "prng_key":
                    leaves[key] = jax.random.wrap_key_data(npx.array(value))
                else:
                    leaves[key] = value if lazy else npx.array(value)
        return leaves
    state = GraphLeaves(_unflatten_state(group), _copy=False)
    return argon.graph.merge(graphdef, state)

import jax.dtypes
def dump(data : Node, store : StoreLike, *, path : StorePath | None = None):
    graphdef, state = argon.graph.split(data)
    group = zarr.create_group(store, path=path, overwrite=True)
    graphdef = _pack_graphdef(graphdef)
    group.attrs["graphdef"] = graphdef
    for k, v in _flatten_state(state):
        v = v.value
        if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key):
            v = np.array(jax.random.key_data(v))
            group.create_array(k, shape=v.shape, dtype=v.dtype)
            group[k].attrs["dtype"] = "prng_key"
            group[k][...] = v
        else:
            v = np.array(v)
            group.create_array(k, shape=v.shape, dtype=v.dtype)
            group[k][...] = v