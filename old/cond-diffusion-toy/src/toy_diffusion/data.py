from argon.core import Array, tree
from argon.core.dataclasses import dataclass, replace
from argon.util.registry import Registry
from argon.datasets.core import Dataset
from argon.data import PyTreeData
from argon.data.normalizer import Identity

import argon.numpy as npx
import argon.random

@dataclass
class Sample:
    x: Array
    y: Array


@dataclass
class InMemory(Dataset[Sample]):
    _data: Sample
    keypoints: Array

    def split(self, name):
        return PyTreeData(self._data)

    def normalizer(self, name, **kwargs):
        if name == "identity":
            return Identity(
                Sample(x=npx.zeros(()), y=npx.zeros(()))
            )
        else:
            raise ValueError(f"Unknown normalizer {name}")


def make_three_deltas():
    data = Sample(x=npx.array([0., 1., 1.]), y=npx.array([-1.0, 0.0, 1.0]))
    data = tree.map(lambda x: npx.repeat(x, 128, axis=0), data)
    data = replace(data, y=data.y + argon.random.normal(argon.random.key(42), data.x.shape) * 0.1)
    return InMemory(data, npx.array([0., 1.]))

def make_seven_deltas():
    data = Sample(x=npx.array([0., 0., 0.5, 0.5, 1., 1., 1.]), y=npx.array([-0.5, 0., -0.1, 0.7, -1.0, 0.0, 1.0]))
    data = tree.map(lambda x: npx.repeat(x, 128, axis=0), data)
    data = replace(data, y=data.y + argon.random.normal(argon.random.key(42), data.x.shape) * 0.1)
    return InMemory(data, npx.array([0., 0.5, 1.]))

def register_all(registry : Registry):
    registry.register("three_deltas", make_three_deltas)
    registry.register("seven_deltas", make_seven_deltas)