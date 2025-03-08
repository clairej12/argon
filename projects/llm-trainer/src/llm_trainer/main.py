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
    iterations: int | None
    dataset: str
    batch_size: int

    learning_rate : float
    weight_decay : float

    def default_options() -> ConfigDict:
        cd = ConfigDict()
        cd.seed = 42
        cd.iterations = None
        cd.dataset = "ops/add_mul_even"
        cd.batch_size = 16
        cd.learning_rate = 1e-4
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
    experiment.log_parameters(dict(opts))

    datasets = Registry()
    register_datasets(datasets)
    dataset = datasets.create(config.dataset)

    logger.info("Loading data...")
    train = PyTreeData(dataset.split("train").as_pytree()[0])
    test = PyTreeData(dataset.split("test").as_pytree()[0])
    prompts = PyTreeData(dataset.split("prompts").as_pytree()[0])

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

    iterations = config.iterations or len(train)

    opt_schedule = optax.warmup_cosine_decay_schedule(
        config.learning_rate/10, config.learning_rate,
        min(int(iterations*0.01), 500), iterations,
        end_value=config.learning_rate/10
    )
    from .gemma.sampler import Sampler

    sampler = Sampler(dataset.tokenizer)

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
        jax.debug.print("{}", positions)
        outputs, _ = model(
            sample_batched, positions, None, attention_mask
        )
        output_logits = npx.squeeze(outputs, 0)

        output_logits = output_logits[9:-1]
        target_labels = sample[10:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            output_logits, target_labels
        ).mean()
        # loss = optax.softmax_cross_entropy_with_integer_labels(
        #     output_logits, sample
        # ).mean()
        return argon.train.LossOutput(
            loss=loss,
            metrics={"loss": loss}
        )

    # Test the loss function
    # loss(model, next(rng), train[0])
    # return
    with prompts.stream().batch(64).shuffle(next(rng)).build() as prompts:
        for step in argon.train.train_model(
                    train.stream().batch(config.batch_size),
                    model, optimizer, loss,
                    iterations=iterations,
                    rng_key=next(rng),
                    store_gradient_variance=True
                ):
            if step.iteration % 10 == 0:
                argon.store.comet.log(experiment, step.iteration, step.metrics, {"grad_var": step.gradient_variance})
            if step.iteration % 100 == 0:
                argon.store.console.log(step.iteration, step.metrics, {"grad_var": step.gradient_variance})
                step.realize()
                if not prompts.has_next():
                    prompts.reset()
                prompt_batch = prompts.next()
                logger.info(f"prompt: {dataset.tokenizer.decode(prompt_batch[0])}")
                samples = sampler(model, prompts=prompt_batch, 
                        total_generation_steps=dataset.generation_steps)
                logger.info(f"sample: {samples.text[0]}")