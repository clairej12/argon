from argon.datasets.common import Dataset
from argon.data.sequence import SequenceData

import argon.numpy as npx
import functools

Token = npx.uint32

class NLPDataset(Dataset):
    def split(self, name: str, **kwargs) -> SequenceData[Token, None]:
        pass

def register(registry):
    from .ops import OperationsDataset
    registry.register("ops/add", functools.partial(
        OperationsDataset, ops=["+"]
    ))
    registry.register("ops/mul", functools.partial(
        OperationsDataset, ops=["*"]
    ))
    registry.register("ops/add_mul_even", functools.partial(
        OperationsDataset, ops=["+", "*"],
        ops_schedule=lambda i, total: npx.array([0.5, 0.5])
    ))
    registry.register("ops/add_mul", functools.partial(
        OperationsDataset, ops=["*", "+"],
        ops_schedule=lambda i, total: npx.array([i/(2*total), 1-i/(2*total)])
    ))
    registry.register("ops/mul_add", functools.partial(
        OperationsDataset, ops=["+", "*"], 
        ops_schedule=lambda i, total: npx.array([i/(2*total), 1-i/(2*total)])
    ))