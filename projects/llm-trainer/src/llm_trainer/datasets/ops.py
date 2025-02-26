from . import NLPDataset, Token
from argon.data.sequence import SequenceData
from argon.data import PyTreeData
from argon.struct import struct, field
from argon.typing import PRNGKey, Array

import argon.transforms as agt
import argon.typing as atp
import argon.random
import argon.numpy as npx

import typing as tp

VOCAB = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '=': 10,
    '+': 11,
    '-': 12,
    '*': 13,
}
OPS = {
    '+': npx.add,
    '-': npx.subtract,
    '*': npx.multiply,
}

@agt.jit
def number_to_tokens(n: atp.Array, digits: int):
    output = []
    for i in range(digits):
        output.append((n // (10**i)) % 10)
    return npx.array(output)

@agt.jit
def generate_op(rng_key: atp.PRNGKey, op_str: str):
    a, b = argon.random.randint(rng_key, (2,), 0, 10000)
    c = OPS[op_str](a, b) % 10000
    a = number_to_tokens(a, 4)
    b = number_to_tokens(b, 4)
    c = number_to_tokens(c, 4)
    op_token = npx.array(VOCAB[op_str])[None]
    return npx.concatenate((a, b, op_token, c))

@agt.jit
def generate_op_from(rng_key: atp.PRNGKey, p, op_strs: str):
    rng_key, p_rng = argon.random.split(rng_key)
    if len(op_strs) == 1:
        return generate_op(rng_key, op_strs[0])
    i = argon.random.choice(p_rng, len(op_strs), p=p)
    return agt.switch(i,
        ((lambda: generate_op(rng_key, op_strs[i])) for i in range(len(op_strs)))
    )

@agt.jit
def generate_ops(rng_key: atp.PRNGKey, n: int, op_schedule: callable, ops: tp.Sequence[str]):
    rng_keys = argon.random.split(rng_key, n)
    idxs = npx.arange(n)
    @agt.vmap
    def _generate_ops(rng_key, i):
        return generate_op_from(rng_key, op_schedule(i, n) if op_schedule is not None else None, ops)
    return _generate_ops(rng_keys, idxs)

@struct
class OperationsDataset(NLPDataset):
    train_rng: PRNGKey = field(default_factory=lambda: argon.random.key(42))
    train_size: int = 100_000
    test_rng: PRNGKey = field(default_factory=lambda: argon.random.key(42))
    test_size: int = 10_000
    ops: tp.Sequence[str] = ()
    ops_schedule: tp.Callable[[int, int], atp.Array] | None = None

    @property
    def vocab_size(self) -> int:
        return len(VOCAB)

    def split(self, name, **kwargs) -> SequenceData[Token, None]:
        if name == "train":
            rng = self.train_rng
            size = self.train_size
        elif name == "test":
            rng = self.test_rng
            size = self.test_size
        else:
            raise ValueError(f"Unknown split {name}")
        data = generate_ops(rng, size, self.ops_schedule, self.ops)
        return SequenceData.from_pytree(data)