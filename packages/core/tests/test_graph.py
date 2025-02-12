import pytest

from argon import graph as agraph

def test_tree_simple():
    tree = (1, 2)
    model = agraph.flatten(tree)
    assert str(model) == "(*, *)"
    assert str(model.pack(tree)) == "Node(0=Leaf(1), 1=Leaf(2))"
    assert tree == model.unpack(model.pack(tree))

    tree = {"a": "foo", "b": "bar"}
    model = agraph.split(tree)
    assert str(model) == "{'a': *, 'b': *}"
    assert str(model.pack(tree)) == "Node(a=Leaf(foo), b=Leaf(bar))"
    assert tree == model.unpack(model.pack(tree))