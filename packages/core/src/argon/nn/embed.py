from flax.nnx import Embed
import flax.nnx as nn

import argon.numpy as npx

class SinusoidalPosEmbed(nn.Module):
    def __init__(self, features):
        self.features = features

    def __call__(self, x):
        x = npx.atleast_1d(x)
        assert x.ndim == 1
        assert self.features % 2 == 0
        half_dim = self.features // 2
        emb = npx.log(10000) / (half_dim - 1)
        emb = npx.exp(npx.arange(half_dim) * -emb)
        x = npx.array(x, dtype=npx.float32)
        emb = x[npx.newaxis,...] * emb[...,npx.newaxis]
        emb = npx.concatenate((npx.sin(emb), npx.cos(emb)), axis=0)
        return emb.reshape((-1))