from zarr.storage import MemoryStore

import argon.numpy as npx
import argon.store
import argon.graph
import argon.random
import chex
import argon.nn

def test_save_load():
    store = MemoryStore()
    data = (1, 2, {"a": npx.ones((10,)), "b": False}, 
            npx.zeros((20,)),
            npx.zeros(()))
    graphdef, state = argon.graph.split(data)
    data = argon.graph.merge(graphdef, state)

    argon.store.dump(data, store)
    loaded_data = argon.store.load(store)
    chex.assert_trees_all_equal(data, loaded_data)

    linear = argon.nn.Linear(10, 10, rngs=argon.nn.Rngs(42))
    _, state = argon.graph.split(linear)

    store = MemoryStore()
    argon.store.dump(state, store)
    loaded_state = argon.store.load(store)
    chex.assert_trees_all_equal(state, loaded_state)
