from __future__ import annotations

import argon.random
import argon.numpy as npx
import argon.typing as atyp
import argon.transforms as agt
import argon.tree
import os

from argon.struct import struct
from argon.data.sequence import Chunk
from argon.datasets.common import DatasetRegistry
from argon.random import PRNGSequence
from argon.envs.common import (
    Environment, ObservationConfig,
    ImageActionsRender
)

# from .methods.behavior_cloning import BCConfig
from .methods.diffusion_policy import DiffusionPolicyTrainer

from .common import DataConfig, Inputs, Method
from .util import setup_logging, parse_options, setup_cache

from functools import partial
from ml_collections import ConfigDict

import functools
import logging
import contextlib
import jax

logger = logging.getLogger(__name__)

@struct(frozen=True)
class Config:
    seed: int
    # these get mixed into the "master" seed
    train_seed: int
    eval_seed: int

    # Dataset configuration
    data_config: DataConfig

    env_timesteps: int
    # if evluate is set, contains a url
    # to the policy to evaluate
    method : Method

    log_compiles: bool

    @staticmethod
    def default_dict() -> ConfigDict:
        cd = ConfigDict()
        cd.seed = 42
        cd.train_seed = 42
        cd.eval_seed = 42
        cd.data = DataConfig.default_dict()
        cd.env_timesteps = 400
        cd.method = "diffusion_policy"
        cd.dp = DiffusionPolicyTrainer.default_dict()
        cd.log_compiles = False
        return cd

    @staticmethod
    def from_dict(dict: ConfigDict) -> Config:
        match dict.method:
          case "diffusion_policy": method = DiffusionPolicyTrainer.from_dict(dict.dp)
          case _: raise ValueError(f"Unknown method {dict.method}")
        return Config(
            seed=dict.seed,
            train_seed=dict.train_seed,
            eval_seed=dict.eval_seed,
            data_config=DataConfig.from_dict(dict.data),
            env_timesteps=dict.env_timesteps,
            method=method,
            log_compiles=dict.log_compiles
        )


@agt.jit
def policy_rollout(env, T, x0, rng_key, policy):
    r_policy, r_env = argon.random.split(rng_key)
    rollout = argon.policy.rollout(
        env.step, x0,
        policy, observe=env.observe,
        model_rng_key=r_env,
        policy_rng_key=r_policy,
        length=T,
        last_action=True
    )
    reward = env.trajectory_reward(rollout.states, rollout.actions)
    return rollout, reward

@agt.jit
def render_video(env, render_width, 
                render_height, rollout):
    states = rollout.states
    actions = rollout.info
    return agt.vmap(env.render)(states, ImageActionsRender(
        render_width, render_height,
        actions=actions
    ))

@agt.jit
def validate(env, T,
             x0s, rng_key, policy) -> atyp.Array:
    rollout_fn = partial(policy_rollout, env, T, policy=policy) 
    # render_fn = partial(render_video, env, 
    #     render_width, render_height
    # )
    N = argon.tree.axis_size(x0s, 0)
    rngs = argon.random.split(rng_key, N)
    rollouts, rewards = agt.vmap(rollout_fn)(x0s, rngs)
    return rewards

def run():
    setup_logging()
    setup_cache()
    logging.getLogger("policy_bench").setLevel(logging.DEBUG)

    dict = Config.default_dict()
    parse_options(dict)
    config = Config.from_dict(dict)
    logger.info(f"Running {config}")
    logger.info(f"Devices: {agt.devices()}")
    jax_context = jax.log_compiles() if config.log_compiles else contextlib.nullcontext()
    with jax_context:
        # ---- set up the random number generators ----

        # split the master RNG into train, eval
        train_key, eval_key = argon.random.split(argon.random.key(config.seed))

        # fold in the seeds for the train, eval
        train_key = argon.random.fold_in(train_key, config.train_seed)
        eval_key = argon.random.fold_in(eval_key, config.eval_seed)

        # ---- set up the training data -----
        dataset = config.data_config.create_dataset()
        env, splits = config.data_config.load(dataset, {"validation"})
        validation_data = splits["validation"].as_pytree()
        N_validation = argon.tree.axis_size(validation_data, 0)

        # validation trajectories
        validate_fn = functools.partial(
            validate, env, config.env_timesteps,
            validation_data
        )

        inputs = Inputs(
            env_timesteps=config.env_timesteps,
            rng=PRNGSequence(train_key),
            env=env,
            dataset=dataset,
            data=config.data_config,
            validate=validate_fn,
        )
        final_result = config.method.run(inputs)
        final_policy = final_result.create_policy()

        logger.info("Running validation for final policy...")
        rewards = validate_fn(eval_key, final_policy)
        mean_reward = npx.mean(rewards)
        std_reward = npx.std(rewards)
        q = npx.array([0, 10, 25, 50, 75, 90, 100])
        quantiles = npx.percentile(rewards, q) 
        outputs = {
            "reward_mean": mean_reward,
            "reward_std": std_reward,
            "reward_quant": {
                f"{q:03}": v for (q, v) in zip(q, quantiles)
            },
        }
        for k, v in outputs.items():
            logger.info(f"{k}: {v}")