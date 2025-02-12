import dataclasses as dcls
import functools
import typing

from dataclasses import MISSING, field, replace

from .tree import NodeModel, Node, model

def fields(cls):
    return dcls.fields(cls)

@typing.dataclass_transform()
def dataclass(identifier=None, version=None,
                frozen=False, kw_only=False, cls=None):
    if (cls is None and identifier is not None and
            not isinstance(identifier, str)):
        cls = identifier
        identifier = None
    if cls is None:
        def decorator(cls):
            return dataclass(
                identifier=identifier,
                version=version,
                frozen=frozen,
                kw_only=kw_only,
                cls=cls
            )
        return decorator
    cls = dcls.dataclass(cls, frozen=frozen, kw_only=kw_only)
    fields = tuple(f.name for f in dcls.fields(cls))

    _identifier = identifier
    _version = version

    class DataclassModel(NodeModel[cls]):
        identifier : str = _identifier
        version : str = _version

        def __init__(self, fields: dict[str, NodeModel]):
            self.fields = fields

        @property
        def children(self) -> dict[str, NodeModel]:
            return self.fields

        def pack_root(self, value, /) -> Node:
            return Node({
                f: getattr(value, f) for f in fields
            })

        def unpack_root(self, node: Node, /):
            return cls(
                **node.children
            )

    # Associate the same model with
    # all instances of the dataclass
    @property
    def __node_model__(self):
        return DataclassModel({
            k: model(getattr(self, k)) for k in fields
        })
    cls.__node_model__ = __node_model__

    return cls
