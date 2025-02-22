from argon.struct import struct, field

import argon.numpy as npx

from argon.typing import Array

@struct
class A:
    x: int
    y: int
    z: int = field(default=0)

@struct(kw_only=True)
class B:
    a: int = None

@struct(kw_only=True)
class C(A):
    foo: Array

def test_simple():
    a = A(1,0)
    assert a.x == 1
    assert a.y == 0
    assert a.z == 0

def test_kwonly():
    b = B(a=1)
    assert b.a == 1