from __future__ import annotations

import argon.transforms as agt
import argon.numpy as npx
import argon.tree
import argon.graph
import argon.random
import argon.typing as atp

from argon.struct import struct, replace
from argon.typing import ArrayLike

from contextlib import contextmanager
from typing import (
    TypeVar, Generic, Callable, Sequence,
    Generator
)

import jax

import math
import numpy as np

T = TypeVar('T')
V = TypeVar('V')

# Make indices 64-bit if x64 is enabled
idx_dtype = int

class DataStream(Generic[T]):
    def has_next(self):
        raise NotImplementedError()

    def next(self) -> T:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
    
    def map(self, fn: Callable[[T], V]) -> DataStream[V]:
        return MappedStream(self, fn)

class StreamBuilder(Generic[T]):
    def batch(self, batch_size: int) -> StreamBuilder[T]:
        raise NotImplementedError()
    
    def shuffle(self, rng_key: jax.Array, resample=False) -> StreamBuilder[T]:
        raise NotImplementedError()
    
    def map(self, fn: Callable[[T], V]) -> StreamBuilder[V]:
        return MappedStreamBuilder(self, fn)
    
    @contextmanager
    def build(self) -> Generator[DataStream[T], None, None]:
        raise NotImplementedError()

class Data(Generic[T]):
    """ A dataset of elements of type T. Not necessarily a jax pytree."""

    # A Data must implement these functions.
    # Non-indexable Data may choose to only implement
    # stream().

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx : ArrayLike) -> T:
        raise NotImplementedError()
    
    def stream(self) -> StreamBuilder[T]:
        return IndexedStreamBuilder(self, len(self))

    # Get the structure of one instance of the data.
    @property
    def structure(self):
        return argon.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self[0])

    def as_pytree(self) -> T:
        idxs = npx.arange(len(self), dtype=idx_dtype)
        return jax.vmap(lambda i: self[i])(idxs)

    def slice(self, off : ArrayLike, length : ArrayLike) -> "Data[T]":
        length = np.array(length).item()
        length = length or len(self) - off
        idxs = npx.arange(length, dtype=idx_dtype) + off
        return PyTreeData(jax.vmap(lambda i: self[i])(idxs))

    def map(self, fn : Callable[[T], V]) -> "MappedData[V]":
        return MappedData(self, fn)

    # "caching" data realizes any transformations
    def cache(self) -> "PyTreeData[T]":
        return PyTreeData(self.as_pytree())
    
@agt.jit
def _compute_mapped_structure(fn, data_structure : T) -> T:
    sample = argon.tree.map(lambda x: npx.zeros(x.shape, x.type), data_structure)
    mapped = fn(sample)
    return argon.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), mapped
    )

@struct(frozen=True)
class MappedData(Data[T]):
    data : Data[V]
    fn: Callable[[V], T]

    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx : ArrayLike) -> T:
        return self.fn(self.data[idx])
    def stream(self) -> StreamBuilder[T]:
        return self.data.stream().map(self.fn)

    @property
    def structure(self) -> T:
        return _compute_mapped_structure(self.fn, self.data.structure)

    def as_pytree(self) -> "T":
        data = self.data.as_pytree()
        return agt.vmap(self.fn)(data)
    
    def slice(self, off : ArrayLike, length : ArrayLike) -> T:
        return self.data.slice(off, length).map(self.fn)

@struct
class MappedStream(DataStream[T]):
    stream: DataStream[V]
    fn: Callable[[V], T]

    def __len__(self):
        return len(self.stream)

    def has_next(self):
        return self.stream.has_next()

    def next(self):
        batch = self.stream.next()
        batch = jax.vmap(self.fn)(batch)
        return batch

    def reset(self):
        return self.stream.reset()

@struct(frozen=True)
class MappedStreamBuilder(StreamBuilder[T]):
    builder: StreamBuilder[V]
    fn: Callable[[V], T]

    def batch(self, batch_size: int) -> "MappedStreamBuilder[T]":
        return MappedStreamBuilder(self.builder.batch(batch_size), self.fn)
    
    def shuffle(self, rng_key : jax.Array, resample=False) -> "MappedStreamBuilder[T]":
        return MappedStreamBuilder(
            self.builder.shuffle(rng_key, resample), self.fn
        )

    @contextmanager
    def build(self) -> Generator[DataStream[T], None, None]:
        with self.builder.build() as stream:
            yield MappedStream(stream, self.fn)

# A Data backed by a jax pytree
class PyTreeData(Data[T]):
    def __init__(self, tree: T | None = None):
        if tree is None:
            self.n = 0
            self.tree = tree
        else:
            with jax.ensure_compile_time_eval():
                ns = npx.array([(npx.shape(x) + (1,))[0] for x in argon.tree.leaves(tree)], dtype=idx_dtype)
                if len(ns) > 0:
                    n = ns[0]
                    assert npx.all(ns == n)
                else:
                    n = 0
            self.n = n
            self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx : ArrayLike) -> T:
        idx = npx.array(idx, dtype=idx_dtype)
        assert idx.ndim == 0
        return argon.tree.map(
            lambda x: x[idx],
            self.tree
        )

    @property
    def structure(self):
        return argon.tree.map(
            lambda x: atp.ShapeDtypeStruct(x.shape[1:], x.dtype),
            self.tree
        )

    def slice(self, off : ArrayLike, length : ArrayLike) -> T:
        # the length must be a scalar
        length = np.array(min(len(self), length)).item()
        return PyTreeData(argon.tree.map(
            lambda x: jax.lax.dynamic_slice(x,
                    npx.broadcast_to(npx.array(off, dtype=idx_dtype), (x.ndim,)),
                    (length,) + x.shape[1:]),
            self.tree
        ))
    
    def as_pytree(self) -> T:
        return self.tree

argon.graph.register_pytree_node_type(
    PyTreeData,
    lambda d: ((("tree", d.tree),), None),
    lambda c, _: PyTreeData(c[0][1])
)

@struct
class IndexedDataStream(DataStream[T]):
    data: Data[T]
    offset: jax.Array
    max_offset: int
    batch_shape: Sequence[int]

    shuffle_key: jax.Array | None
    indices: jax.Array | None
    resample : bool

    @staticmethod
    def create(data, max_offset, batch_shape,
               shuffle_key=None, resample=False, ):
        indices_per_batch = math.prod(batch_shape)
        if indices_per_batch > max_offset: 
            # reduce batch_shape to fit at least one batch
            batch_rem = math.prod(batch_shape[1:])
            leading_axis = max_offset // batch_rem
            if leading_axis > 0:
                batch_shape = (leading_axis,) + tuple(batch_shape[1:])
                indices_per_batch = math.prod(batch_shape)

        batches = max_offset // indices_per_batch
        max_offset = batches * indices_per_batch
        if shuffle_key is not None and not resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            indices = jax.random.permutation(r, max_offset)
        else: indices = None
        return IndexedDataStream(
            data=data,
            offset=npx.zeros((), dtype=idx_dtype),
            max_offset=max_offset,
            batch_shape=batch_shape,
            shuffle_key=shuffle_key,
            indices=indices,
            resample=resample,
        )

    def __len__(self):
        batch_size = math.prod(self.batch_shape)
        return (self.max_offset - self.offset) // batch_size

    @agt.jit
    def has_next(self):
        return self.offset < self.max_offset

    @agt.jit
    def next(self):
        shuffle_key = self.shuffle_key
        batch_shape = self.batch_shape
        batch_size = math.prod(batch_shape)
        if self.resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            idxs = jax.random.randint(r, (batch_size,), minval=0, maxval=self.max_offset)
            data = jax.vmap(lambda x: self.data[x])(idxs)
        elif self.indices is not None:
            idxs = jax.lax.dynamic_slice(self.indices, self.offset[None], self.batch_shape)
            data = jax.vmap(lambda i: self.data[i])(idxs)
        else:
            data = self.data.slice(self.offset, batch_size).as_pytree()
        data = argon.tree.map(lambda x: npx.reshape(x, batch_shape + x.shape[1:]), data)

        self.offset = self.offset + batch_size
        self.shuffle_key = shuffle_key

        return data
    
    @agt.jit
    def reset(self):
        self.offset = npx.zeros_like(self.offset)
        self.indices = None
        if self.shuffle_key is not None and not self.resample:
            shuffle_key, r = argon.random.split(self.shuffle_key)
            self.indices = argon.random.permutation(r, self.max_offset)
            self.shuffle_key = shuffle_key

@struct(frozen=True)
class IndexedStreamBuilder(StreamBuilder[T]):
    data: Data[T]
    max_offset: int
    batch_shape: Sequence[int] | None = None
    shuffle_key: jax.Array | None = None
    resample : bool = False

    def batch(self, batch_size: int) -> "IndexedStreamBuilder[T]":
        return replace(self, 
            batch_shape=((batch_size,) + self.batch_shape) 
            if self.batch_shape else (batch_size,)
        )
    
    def shuffle(self, rng_key : jax.Array, resample=False) -> "IndexedStreamBuilder[T]":
        return replace(self,
            shuffle_key=rng_key if self.shuffle_key is None else argon.random.fold_in(self.shuffle_key, rng_key),
            resample=resample or self.resample
        )

    @contextmanager
    def build(self) -> Generator[DataStream[T], None, None]:
        yield IndexedDataStream.create(
            self.data, self.max_offset, self.batch_shape,
            self.shuffle_key, self.resample
        )