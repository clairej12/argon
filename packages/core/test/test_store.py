from zarr.storage import MemoryStore

import argon.numpy as npx
import argon.store
import argon.graph
import chex

def test_save_load():
    store = MemoryStore()
    data = (1, 2, {"a": npx.ones((10,)), "b": False}, npx.zeros((20,)))

    graphdef, state = argon.graph.split(data)
    data = argon.graph.merge(graphdef, state)

    argon.store.dump(data, store)
    loaded_data = argon.store.load(store)
    chex.assert_trees_all_equal(data, loaded_data)