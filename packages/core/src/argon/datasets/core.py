from typing import Generic, TypeVar, Mapping, Callable

from argon.core.dataclasses import dataclass, field
from argon.data import Data
from argon.data.transform import Transform
from argon.data.normalizer import Normalizer

from argon.util.registry import Registry

import logging
logger = logging.getLogger("argon.datasets")

T = TypeVar('T')

@dataclass
class Dataset(Generic[T]):
    def split(self, name : str) -> Data[T] | None:
        return None

    def augmentation(self, name : str, **kwargs) -> Transform | None:
        return None

    def normalizer(self, name : str, **kwargs) -> Normalizer[T] | None:
        return None

DatasetRegistry = Registry

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
