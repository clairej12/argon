from argon.data import PyTreeData
from argon.models.mlp import MLP

import argon.numpy as npx
import argon.train
import argon.random
import argon.nn as nn
import optax

def test_simple():
    data = PyTreeData(
        {
            "x": npx.zeros((3*16,)), 
            "y": npx.ones((3*16,))
         }
    )

    with argon.train.loop(
                data.stream().batch(16).shuffle(argon.random.key(42)),
                iterations=100,
            ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                pass

def test_train():
    data = PyTreeData(
        {
            "x": npx.zeros((3*16,1)), 
            "y": npx.ones((3*16,1))
         }
    )
    model = MLP(1, 1, [16, 16], rngs=nn.Rngs(42))
    optimizer = nn.Optimizer(model, optax.adam(1e-3))

    def loss(model, rng_key, sample):
        pred = model(sample["x"])
        loss = npx.mean((pred - sample["y"])**2)
        return argon.train.LossOutput(
            loss=loss, metrics={"mse": loss}
        )

    for step in argon.train.train_model(
                data.stream().batch(16), 
                model, optimizer, loss, 
                iterations=10
            ):
        # This will update the model, optimizer
        # but is costly and therefore should be avoided if possible
        step.realize()
        pass