import dataclasses as dcls
import functools
import typing

import argon.tree
import argon.typing as atyp

from argon.nn import Variable
from dataclasses import MISSING, field, replace
from typing import Callable, TypeVar, overload
from . import graph

_T = TypeVar('_T')

@typing.dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@overload
def struct(clz: _T, **kwargs) -> _T:
  ...

@typing.dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@overload
def struct(**kwargs) -> Callable[[_T], _T]:
  ...

@typing.dataclass_transform()
def struct(cls=None, frozen=False, kw_only=False):
    if cls is None:
        return functools.partial(struct, 
            frozen=frozen, kw_only=kw_only
        )
    cls = dcls.dataclass(cls, frozen=frozen, kw_only=kw_only)
    fields = tuple(f.name for f in dcls.fields(cls))

    if frozen:
        def flatten(x):
            children  = {f: getattr(x, f) for f in fields}
            children = sorted(children.items())
            return children, None
        def unflatten(children, _):
            o = cls.__new__(cls)
            for f, v in children:
                object.__setattr__(o, f, v)
            return o
        graph.register_pytree_node_type(
            cls, flatten, unflatten
        )
    else:
        def flatten(x):
            children  = {f: getattr(x, f) for f in fields}
            children = argon.tree.map(
                lambda x: x,
                children
            )
            children = sorted(children.items())
            return children, None

        def set_key(x, key, value):
            value = argon.tree.map(
                lambda x: x,
                value
            )
            setattr(x, key, value)
        
        def pop_key(x, key):
            value = getattr(x, key)
            setattr(x, key, MISSING)
            return value
        
        def create_empty(_):
            return cls.__new__(cls)

        def clear(x):
            for f in fields:
                setattr(x, f, MISSING)
        
        def init(x, keys):
            for k, v in keys:
                object.__setattr__(x, k, v)

        graph.register_graph_node_type(
          cls, flatten,
          set_key, pop_key,
          create_empty, clear, init
        )

    return cls