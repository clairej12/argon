from .util import setup_logging, setup_cache, parse_options
from .datasets import register as register_datasets

from argon.registry import Registry
from argon.struct import struct
from argon.data import PyTreeData

import argon.store.console
import argon.store.comet

import argon.tree
import argon.graph
import argon.train
import argon.random
import argon.numpy as npx
import argon.transforms as agt
import argon.nn as nn

import jax.debug

import typing as tp
import optax

import comet_ml
from ml_collections import ConfigDict


import sys
import logging
logger = logging.getLogger(__name__)


@struct
class Config:
    seed: int
    iterations: int
    dataset: str
    batch_size: int

    learning_rate : float
    weight_decay : float

    def default_options() -> ConfigDict:
        cd = ConfigDict()
        cd.seed = 42
        cd.iterations = 10_000
        cd.dataset = "ops/add_mul_even"
        cd.batch_size = 16
        cd.learning_rate = 3e-4
        cd.weight_decay = 1e-4
        return cd
    
    @staticmethod
    def from_options(opts: ConfigDict) -> tp.Self:
        return Config(
            seed=opts.seed,
            iterations=opts.iterations,
            dataset=opts.dataset,
            batch_size=opts.batch_size,
            learning_rate=opts.learning_rate,
            weight_decay=opts.weight_decay
        )

def run():
    setup_logging()
    setup_cache()
    logger.setLevel(logging.DEBUG)
    opts = Config.default_options()
    parse_options(opts, sys.argv[1:])
    config = Config.from_options(opts)
    rng = argon.random.sequence(config.seed)
    logger.info(f"Configuration: {config}")

    experiment = comet_ml.start(project_name="llm-trainer")

    datasets = Registry()
    register_datasets(datasets)
    dataset = datasets.create(config.dataset)

    logger.info("Loading data...")
    train = PyTreeData(dataset.split("train").as_pytree()[0])
    test = PyTreeData(dataset.split("test").as_pytree()[0])

    logger.info("Creating model...")
    from .gemma.transformer import Transformer, TransformerConfig
    from .gemma import transformer as transformerlib
    model = Transformer(
        TransformerConfig.gemma_nano(
            dataset.vocab_size
        ),
        rngs=nn.Rngs(next(rng))
    )
    _, vars = argon.graph.split(model)
    total_vars = argon.tree.reduce(npx.add, 
            argon.tree.map(lambda x: x.size, vars))
    logger.info(f"Total: {total_vars} parameters")

    opt_schedule = optax.warmup_cosine_decay_schedule(
        config.learning_rate/10, config.learning_rate,
        min(int(config.iterations*0.01), 500), config.iterations,
        end_value=config.learning_rate/10
    )

    optimizer = nn.Optimizer(model, optax.adamw(opt_schedule,
        weight_decay=config.weight_decay
    ))

    @agt.jit
    def loss(model : Transformer, rng_key, sample):
        # Add a batch axis to the sample
        sample_batched = sample[None, ...]
        input_mask = npx.ones(sample_batched.shape, dtype=npx.bool)
        positions = transformerlib.build_positions_from_mask(input_mask)
        attention_mask = transformerlib.make_causal_attn_mask(input_mask)
        outputs, _ = model(
            sample_batched, positions, None, attention_mask
        )
        output_logits = npx.squeeze(outputs, 0)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            output_logits[:-1], sample[1:]
        ).mean()
        return argon.train.LossOutput(
            loss=loss,
            metrics={"loss": loss}
        )

    # Test the loss function
    # loss(model, next(rng), train[0])
    # return
    for step in argon.train.train_model(
                train.stream().batch(config.batch_size),
                model, optimizer, loss,
                iterations=config.iterations,
                rng_key=next(rng),
                store_gradient_variance=True
            ):
        argon.store.comet.log(experiment, step.iteration, step.metrics, {"grad_var": step.gradient_variance})
        if step.iteration % 100 == 0:
            argon.store.console.log(step.iteration, step.metrics, {"grad_var": step.gradient_variance})