# from argon.dataclasses import dataclass, field

# import argon.numpy as npx
# import argon.tree as agtree

# from argon.typing import Array

# @dataclass
# class A:
#     x: int
#     y: int
#     z: int = field(default=0)

# @dataclass(kw_only=True)
# class B:
#     a: int = None

# @dataclass(kw_only=True)
# class C(A):
#     foo: Array

# def test_simple():
#     a = A(1,0)
#     assert a.x == 1
#     assert a.y == 0
#     assert a.z == 0

# def test_kwonly():
#     b = B(a=1)
#     assert b.a == 1

# def test_tree():
#     c = C(x=0, y=1, z=2, foo=1)
#     tree = agtree.Tree(c)
#     assert tree.unpack() == c
