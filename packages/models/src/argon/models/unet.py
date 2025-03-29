import argon.nn as nn
import argon.numpy as npx
import argon.typing as atp
import argon.transforms as agt

import math
import typing as typ

from argon.registry import Registry

from .resnet import (
    ConvDef, NormDef, ActivationDef,
    Embed,
    CondSequential,
    ResNetBlock
)
import functools

def _qkv_attention(q,k,v, n_heads):
    length, channels = q.shape
    assert q.shape == k.shape == v.shape
    ch = channels // n_heads
    scale = 1 / npx.sqrt(npx.sqrt(ch))
    weight = npx.einsum(
        "thc,shc->ths",
        (q*scale).reshape(length, n_heads, ch),
        (k*scale).reshape(length, n_heads, ch),
    )
    weight = nn.activations.softmax(weight, axis=-1)
    a = npx.einsum(
        "ths,shc->thc",
        weight,
        v.reshape(length, n_heads, ch),
    )
    return a

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, spatial_dims: int,
                 num_heads: int | None = None,
                 num_head_channels: int | None = None, *,
                 norm: NormDef = nn.GroupNorm,
                 rngs: nn.Rngs):
        self.norm = norm(channels, rngs=rngs)
        self.qkv_conv = nn.Linear(channels, 3*channels, rngs=rngs)
        self.num_heads = num_heads or channels // num_head_channels

    def __call__(self, x : atp.Array):
        spatial_shape = x.shape[:-1]

        x = x[None, ...]
        x = self.norm(x)
        x = npx.squeeze(x, axis=0)

        T = math.prod(spatial_shape)
        F = x.shape[-1]
        x = x.reshape(T, F)
        x = agt.vmap(self.qkv_conv)(x)
        x = x.reshape(T, F, 3)
        q,k,v = x[..., 0], x[...,1], x[...,2]
        a = _qkv_attention(q,k,v, self.num_heads)
        a = npx.reshape(a, spatial_shape + (a.shape[-1],))
        return a

class UNet(nn.Module):
    def __init__(self, in_channels : int, out_channels : int,
                model_channels : int, 
                *,
                kernel_size : int | typ.Sequence[int] = 3,
                spatial_dims: int | None = None,

                embed: Embed | None = None,
                embed_features: int | None = None,

                num_heads_downsample: int = 1,
                num_heads_upsample: int = 1,
                channel_mults: typ.Sequence[int] = (2, 4, 8),
                attention_resolutions: typ.Sequence[int] = (4,8,),
                blocks_per_level: int | typ.Sequence[int] = 2,
                dropout: float | None = 0.5,
                use_film: bool = True,
                use_resblock_updown: bool = False,
                activation: ActivationDef = nn.activations.silu,
                conv: ConvDef = nn.Conv,
                norm: NormDef = nn.GroupNorm,
                rngs: nn.Rngs):
        if isinstance(kernel_size, int):
            if spatial_dims is None:
                raise ValueError("spatial_dims must be provided if kernel_size is an int")
            kernel_size = (kernel_size,) * spatial_dims
        elif spatial_dims is not None:
            assert len(kernel_size) == spatial_dims
        else:
            spatial_dims = len(kernel_size)

        if isinstance(blocks_per_level, int):
            blocks_per_level = (blocks_per_level,) * len(channel_mults)
        else:
            assert len(blocks_per_level) == len(channel_mults)
        ResBlock = functools.partial(ResNetBlock,
                    kernel_size=kernel_size, dropout=dropout, cond_features=embed_features,
                    use_film=use_film, activation=activation, conv=conv, norm=norm, rngs=rngs)
        AttenBlock = functools.partial(AttentionBlock, spatial_dims=spatial_dims, rngs=rngs)

        self.embed = embed
        self.input_conv = conv(in_channels, model_channels, kernel_size, rngs=rngs)

        self.input_blocks = []
        skip_channels = []
        ds = 1
        for level, (prev_mult, mult, num_blocks) in enumerate(zip(
                (1,) + channel_mults[:-1],
                channel_mults, blocks_per_level)):
            level_in_channels = int(prev_mult*model_channels)
            level_out_channels = int(mult*model_channels)
            level_blocks = []
            for i in range(num_blocks):
                block_in_channels = level_in_channels if i == 0 else level_out_channels
                level_blocks.append(ResBlock(block_in_channels, level_out_channels,))
                if ds in attention_resolutions:
                    level_blocks.append(AttenBlock(level_out_channels, num_heads=num_heads_downsample))
            if level < len(channel_mults) - 1:
                if use_resblock_updown:
                    level_blocks.append(ResBlock(level_out_channels, level_out_channels,
                                                 operation=nn.downsample))
                else:
                    level_blocks.append(nn.downsample)
                ds *= 2
            self.input_blocks.append(CondSequential(*level_blocks))
            skip_channels.append(level_out_channels)

        middle_channels = int(channel_mults[-1]*model_channels)
        self.middle_block = CondSequential(
            ResBlock(middle_channels, middle_channels),
            AttenBlock(middle_channels, num_heads=num_heads_downsample),
            ResBlock(middle_channels, middle_channels)
        )

        self.output_blocks = []
        for level, (prev_mult, mult, num_blocks, skip_ch) in enumerate(zip(
                    reversed(channel_mults),
                    reversed((1,) + channel_mults[:-1]),
                    reversed(blocks_per_level), reversed(skip_channels)
                )):
            level_blocks = []
            level_in_channels = int(prev_mult*model_channels) + skip_ch
            level_out_channels = int(mult*model_channels)
            for i in range(num_blocks):
                block_in_channels = level_in_channels if i == 0 else level_out_channels
                level_blocks.append(ResBlock(block_in_channels, level_out_channels))
                if ds in attention_resolutions:
                    level_blocks.append(AttenBlock(level_out_channels, num_heads=num_heads_upsample))
            if level > 0:
                if use_resblock_updown:
                    level_blocks.append(ResBlock(level_out_channels, level_out_channels,
                                                 operation=nn.upsample))
                else:
                    level_blocks.append(nn.upsample)
                ds //= 2
            self.output_blocks.append(CondSequential(*level_blocks))
        self.out_final = nn.Sequential(
            lambda x: x[None, ...],
            norm(model_channels, rngs=rngs),
            lambda x: npx.squeeze(x, axis=0),
            activation,
            conv(model_channels, out_channels, kernel_size, rngs=rngs)
        )

    def __call__(self, x, cond=None):
        if self.embed is not None and cond is not None:
            cond = self.embed(cond)
        x = self.input_conv(x)
        skip_xs = []
        target_shapes = []
        for block in self.input_blocks:
            target_shapes.append(x.shape[:-1])
            x = block(x, cond=cond)
            skip_xs.append(x)
        x = self.middle_block(x, cond=cond)
        for block in self.output_blocks:
            skip_x = skip_xs.pop()
            target_shape = target_shapes.pop()
            x = npx.concatenate([x, skip_x], axis=-1)
            x = block(x, cond=cond, target_shape=target_shape)
        x  = self.out_final(x)
        return x
import jax.debug
def register(registry: Registry[nn.Module]):
    from functools import partial
    MicroUNet = partial(UNet, model_channels=8, channel_mults=(2,4))
    SmallUNet = partial(UNet, model_channels=16, channel_mults=(2,4,8))
    MediumUNet = partial(UNet, model_channels=32, channel_mults=(2,4,8))
    LargeUNet = partial(UNet, model_channels=64, channel_mults=(2,4,8))

    registry.register("unet", UNet)
    registry.register("unet/micro", MicroUNet)
    registry.register("unet/small", SmallUNet)
    registry.register("unet/medium", MediumUNet)
    registry.register("unet/large", LargeUNet)
