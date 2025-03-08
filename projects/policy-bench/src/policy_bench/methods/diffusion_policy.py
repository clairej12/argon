import argon.random

from argon.struct import struct
from argon.diffusion import DDPMSchedule
from argon.random import PRNGSequence
from argon.policy import Policy, PolicyInput, PolicyOutput, ChunkingTransform

from argon.graph import GraphLeaves

from argon.data.normalizer import Normalizer, LinearNormalizer, StdNormalizer
from argon import train

import argon.models.mlp as mlp
import argon.store.console
import argon.store.comet
import argon.tree
import argon.nn as nn
import argon.numpy as npx
import argon.typing as atp
import argon.transforms as agt

from argon.nn.embed import SinusoidalPosEmbed
from argon.models.unet import UNet, ActivationDef

from ml_collections import FrozenConfigDict, ConfigDict

import jax.tree_util
import typing as tp
import chex
import optax
import logging

from ..common import Sample, Inputs, Result, DataConfig, Method
from argon.envs.common import Action, Observation

logger = logging.getLogger(__name__)

class ModelConfig: ...

class UNetDiffuser(tp.Generic[Action, Observation], nn.Module):
    def __init__(self,
                observations_structure: Observation,
                actions_structure: Action,
                cond_embed_dim=64, activation: ActivationDef = nn.activations.silu,
                model_channels=32,
                *, rngs : nn.Rngs
            ):
        self.actions_structure = actions_structure
        self.observations_structure = observations_structure

        actions_flat = agt.vmap(lambda x: argon.tree.ravel_pytree(x)[0])(
            argon.tree.map(lambda x: npx.zeros_like(x), actions_structure)
        )
        observations_flat = argon.tree.ravel_pytree(
            argon.tree.map(lambda x: npx.zeros_like(x), observations_structure)
        )[0]
        self.unet = UNet(
            in_channels=actions_flat.shape[-1],
            out_channels=actions_flat.shape[-1],
            model_channels=model_channels,
            spatial_dims=1,
            embed_features=cond_embed_dim, rngs=rngs
        )
        self.time_embed = SinusoidalPosEmbed(cond_embed_dim)
        self.obs_embed = nn.Sequential(
            nn.Linear(observations_flat.shape[-1], 2*observations_flat.shape[-1], rngs=rngs),
            activation,
            nn.Linear(2*observations_flat.shape[-1], cond_embed_dim, rngs=rngs),
        )

    @agt.jit
    def __call__(self, observations, rng_key, actions, t):
        chex.assert_trees_all_equal_shapes_and_dtypes(actions, self.actions_structure)
        chex.assert_trees_all_equal_shapes_and_dtypes(observations, self.observations_structure)

        _, action_uf = argon.tree.ravel_pytree(argon.tree.map(lambda x: x[0], actions))
        actions_uf = agt.vmap(lambda x: action_uf(x))
        actions_flat = agt.vmap(lambda x: argon.tree.ravel_pytree(x)[0])(actions)
        observations_flat = argon.tree.ravel_pytree(observations)[0]

        cond_embed = self.obs_embed(observations_flat) + self.time_embed(t)
        actions_flat = self.unet(actions_flat, cond_embed)
        return actions_uf(actions_flat)

class MlpDiffuser(nn.Module):
    def __init__(self, observations_structure: Observation,
                 actions_structure: Action, *, rngs: nn.Rngs):
        self.actions_structure = actions_structure
        self.observations_structure = observations_structure

        actions_flat = argon.tree.ravel_pytree(
            argon.tree.map(lambda x: npx.zeros_like(x), actions_structure)
        )[0]
        observations_flat = argon.tree.ravel_pytree(
            argon.tree.map(lambda x: npx.zeros_like(x), observations_structure)
        )[0]
        self.mlp = mlp.MLP(
            actions_flat.shape[0] + observations_flat.shape[0] + 1, 
            actions_flat.shape[0], hidden_features=[32, 32, 32],
            activation=nn.activations.gelu, rngs=rngs
        )

    @agt.jit
    def __call__(self, observations, rng_key, actions, t):
        chex.assert_trees_all_equal_shapes_and_dtypes(actions, self.actions_structure)
        chex.assert_trees_all_equal_shapes_and_dtypes(observations, self.observations_structure)
        actions_flat, actions_uf = argon.tree.ravel_pytree(actions)
        observations_flat = argon.tree.ravel_pytree(observations)[0]

        actions_flat = self.mlp(
            npx.concatenate((actions_flat, observations_flat, t[None]), axis=0)
        )
        return actions_uf(actions_flat)

@struct(frozen=True)
class UNetConfig:
    @staticmethod
    def default_dict() -> ConfigDict:
        return ConfigDict()

    @staticmethod
    def from_dict(config: FrozenConfigDict) -> tp.Self:
        return UNetConfig()

    def create_model(self, rng_key, obs_structure, action_structure) -> nn.Module:
        rngs = nn.Rngs(rng_key)
        model = UNetDiffuser(
            obs_structure, action_structure,
            rngs=rngs
        )
        return model
    
    def load_model(self, model_state : GraphLeaves, 
                    obs_structure, action_structure) -> nn.Module:
        abstract_model = agt.eval_shape(lambda: UNetDiffuser(
            obs_structure, action_structure, rngs=nn.Rngs(42)
        ))
        graphdef, _ = argon.graph.split(abstract_model)
        return argon.graph.merge(graphdef, model_state)

@struct(frozen=True)
class MlpConfig:
    @staticmethod
    def default_dict() -> ConfigDict:
        return ConfigDict()

    @staticmethod
    def from_dict(config: FrozenConfigDict) -> tp.Self:
        return MlpConfig()

    def create_model(self, rng_key, obs_structure, action_structure) -> nn.Module:
        rngs = nn.Rngs(rng_key)
        model = MlpDiffuser(
            obs_structure, action_structure,
            rngs=rngs
        )
        return model
    
    def load_model(self, model_state : GraphLeaves, 
                    obs_structure, action_structure) -> nn.Module:
        abstract_model = agt.eval_shape(lambda: MlpDiffuser(
            obs_structure, action_structure, rngs=nn.Rngs(42)
        ))
        graphdef, _ = argon.graph.split(abstract_model)
        return argon.graph.merge(graphdef, model_state)

@struct(frozen=True)
class DiffusionPolicyTrainer(Method):
    model_config: ModelConfig

    epochs: int | None
    iterations : int
    batch_size: int
    learning_rate: float
    weight_decay: float
    replica_noise: float | None

    diffusion_steps: int
    action_horizon: int

    log_video: bool

    @staticmethod
    def default_dict() -> ConfigDict:
        cd = ConfigDict()
        cd.unet  = UNetConfig.default_dict()
        cd.mlp = MlpConfig.default_dict()
        cd.model = "unet"
        cd.epochs = None
        cd.iterations = 10_000
        cd.batch_size = 64
        cd.learning_rate = 3e-4
        cd.weight_decay = 1e-5
        cd.replica_noise = 0.
        cd.diffusion_steps = 32
        cd.action_horizon = 8
        cd.log_video = False
        return cd

    @staticmethod
    def from_dict(config: FrozenConfigDict) -> tp.Self:
        match config.model:
            case "unet": model_config = UNetConfig.from_dict(config.unet)
            case "mlp": model_config = MlpConfig.from_dict(config.mlp)
            case _: raise ValueError(f"Unknown model {config.model}")
        return DiffusionPolicyTrainer(
            model_config=model_config,
            epochs=config.epochs,
            iterations=config.iterations,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            replica_noise=config.replica_noise,
            diffusion_steps=config.diffusion_steps,
            action_horizon=config.action_horizon,
            log_video=config.log_video
        )

    def run(self, inputs: Inputs):
        action_horizon = min(self.action_horizon, inputs.data.action_length)
        train_rng, shuffle_rng, val_rng = next(inputs.rng), next(inputs.rng), next(inputs.rng)

        _, data = inputs.data.load(inputs.dataset, {"train", "test"})
        logger.info("Materializing all data...")
        train_data = data["train"].cache()

        schedule = DDPMSchedule.make_squaredcos_cap_v2(
            self.diffusion_steps,
            prediction_type="epsilon",
            clip_sample_range=1.0,
        )
        obs_structure, action_structure = (
            argon.tree.structure(train_data[0].observations),
            argon.tree.structure(train_data[0].actions)
        )
        logger.info(f"Observation: {obs_structure}")
        logger.info(f"Action: {action_structure}")
        model = self.model_config.create_model(
            next(inputs.rng), obs_structure, action_structure,
        )
        total_params = sum(v.size for v in argon.tree.leaves(argon.graph.split(model)[1]))

        logger.info(f"Total parameters: {total_params}")

        def make_checkpoint(model) -> DiffusionPolicy:
            model_state = argon.graph.split(model)[1]
            return DiffusionPolicy(
                data=inputs.data,
                observations_structure=obs_structure,
                actions_structure=action_structure,
                model_config=self.model_config,
                model_state=model_state,
                action_horizon=action_horizon,
                schedule=schedule,
                obs_normalizer=normalizer.map(lambda x: x.observations),
                action_normalizer=normalizer.map(lambda x: x.actions),
            )

        # normalizer = StdNormalizer.from_data(train_data)
        normalizer = LinearNormalizer.from_data(train_data)

        epoch_iterations = len(train_data) // self.batch_size
        if self.epochs is not None:
            total_iterations = self.epochs * epoch_iterations
        elif self.iterations is not None:
            total_iterations = self.iterations
        else:
            raise ValueError("Must specify either epochs or iterations")

        # initialize optimizer, EMA
        opt_schedule = optax.warmup_cosine_decay_schedule(
            self.learning_rate/10, self.learning_rate,
            min(int(total_iterations*0.01), 500), total_iterations,
            end_value=self.learning_rate/10
        )
        optimizer = nn.Optimizer(model, optax.adamw(opt_schedule,
            weight_decay=self.weight_decay
        ))

        @agt.jit
        def loss_fn(model : nn.Module, rng_key, sample: Sample):
            # re-build the schedule to get around issues with flax nnx
            schedule = DDPMSchedule.make_squaredcos_cap_v2(
                self.diffusion_steps,
                prediction_type="sample"
            )
            sample = normalizer.normalize(sample)
            actions, obs = sample.actions, sample.observations
            denoiser = lambda rng_key, noised_actions, t: model(
                obs, rng_key, noised_actions, t
            )
            loss = schedule.loss(rng_key, denoiser, actions)
            return train.LossOutput(
                loss=loss, metrics={"loss": loss}
            )

        # validate = agt.jit(
        #     lambda model: inputs.validate(
        #         val_rng, make_checkpoint(model).create_policy()
        #     )
        # )
        for step in train.train_model(train_data.stream().batch(self.batch_size).shuffle(shuffle_rng), model, optimizer, loss_fn,
                               iterations=total_iterations, rng_key=train_rng):
            argon.store.comet.log(inputs.experiment, step.iteration, step.metrics)
            if step.iteration % 100 == 0:
                argon.store.console.log(step.iteration, step.metrics)
                argon.store.console.log(step.iteration, step.sys_metrics, prefix="sys.")
        return make_checkpoint(model)

@struct
class DiffusionPolicy(Result):
    data: DataConfig # dataset this model was trained on
    observations_structure: Observation
    actions_structure: Action
    model_config: ModelConfig
    model_state: GraphLeaves
    schedule: DDPMSchedule
    action_horizon: int

    obs_normalizer: Normalizer
    action_normalizer: Normalizer

    def create_denoiser(self):
        return self.model_config.load_model(self.model_state, 
            self.observations_structure, self.actions_structure)

    def create_policy(self) -> Policy:
        actions_structure = self.actions_structure
        observations_structure = self.observations_structure
        model = self.model_config.load_model(self.model_state, 
                    observations_structure, actions_structure)
        action_horizon = self.action_horizon
        def chunk_policy(self, model, input: PolicyInput) -> PolicyOutput:
            s_rng, n_rng = argon.random.split(input.rng_key)
            obs = input.observation
            obs = self.obs_normalizer.normalize(obs)
            def denoiser(model, obs, rng_key, actions, t):
                obs = model(obs, rng_key, actions, t)
                obs_flat, obs_uf = argon.tree.ravel_pytree(obs)
                obs_flat = obs_flat.clip(-2, 2)
                return obs_uf(obs_flat)
            denoiser = agt.partial(denoiser, model, obs)
            action = self.schedule.sample(s_rng, denoiser, actions_structure)
            action = self.action_normalizer.unnormalize(action)
            action = argon.tree.map(lambda x: x[:action_horizon], action)
            return PolicyOutput(action=action, info=action)
        chunk_policy = agt.partial(chunk_policy, self, model)
        obs_horizon = argon.tree.axis_size(observations_structure, 0)
        return ChunkingTransform(
            obs_horizon, action_horizon
        ).apply(chunk_policy)