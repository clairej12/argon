from argon.datasets.common import Dataset
from argon.data.sequence import SequenceData
from argon.struct import struct

import typing as typ

T = typ.TypeVar('T')
E = typ.TypeVar('E')

@struct(frozen=True)
class Step:
    # either state or reduced_state must be set
    state: typ.Any | None
    reduced_state: typ.Any | None
    observation: typ.Any
    action: typ.Any

class EnvDataset(typ.Generic[T, E], Dataset[T]):
    def split(name, **kwargs) -> SequenceData[T, None]:
        return None

    def env(self, **kwargs) -> E:
        raise NotImplementedError()