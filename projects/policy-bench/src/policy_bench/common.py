import argon.transforms as agt
import argon.numpy as npx
import argon.typing as atyp
import argon.tree

from argon.policy import Rollout
from argon.random import PRNGSequence
from argon.envs.common import Environment, ObservationConfig

from argon.datasets.common import DatasetRegistry
from argon.datasets.envs.common import EnvDataset, Step
from argon.datasets.envs import pusht as pusht_datasets

from argon.struct import struct
from argon.policy import Policy
from argon.data import Data

from .envs import lower_bounds as lower_bounds

import typing as tp
from ml_collections import ConfigDict

import comet_ml
import logging

logger = logging.getLogger(__name__)

@struct(frozen=True)
class Sample:
    state: tp.Any
    observations: atyp.Array
    actions: atyp.Array
    rel_actions: atyp.Array

@struct(frozen=True)
class DataConfig:
    dataset: str
    observation_type: str | None

    action_length: int
    obs_length: int

    train_trajectories: int | None
    test_trajectories: int | None
    validation_trajectories: int | None

    @staticmethod
    def default_dict() -> ConfigDict:
        cd = ConfigDict()
        cd.action_length = 8
        cd.obs_length = 1
        cd.dataset = "pusht/chi"
        cd.observation_type = "keypoint"
        cd.train_trajectories = None
        cd.test_trajectories = None
        cd.validation_trajectories = None
        return cd

    @staticmethod
    def from_dict(dict: ConfigDict) -> tp.Self:
        return DataConfig(
            dataset=dict.dataset,
            observation_type=dict.observation_type,
            train_trajectories=dict.train_trajectories,
            test_trajectories=dict.test_trajectories,
            validation_trajectories=dict.validation_trajectories,
            action_length=dict.action_length,
            obs_length=dict.obs_length
        )

    def _process_data(self, env : Environment, data):
        def process_element(element : Step):
            if element.state is None: 
                return (env.full_state(element.reduced_state), element.action)
            else: return (element.state, element.action)
        data = data.map_elements(process_element).cache()
        data = data.chunk(
            self.action_length + self.obs_length - 1
        )
        @agt.jit
        def process_chunk(chunk):
            from argon.envs.pusht import PushTAgentPos
            states, actions = chunk.elements
            actions = argon.tree.map(lambda x: x[-self.action_length:], actions)
            obs_states = argon.tree.map(lambda x: x[:self.obs_length], states)
            curr_state = argon.tree.map(lambda x: x[-1], obs_states)
            last_obs = env.observe(curr_state)
            obs = agt.vmap(env.observe)(obs_states)
            actions_relative = agt.vmap(
                lambda actions: self.relative_action(last_obs, actions), 
            )(actions)
            return Sample(
                curr_state, obs, actions,
                actions_relative
            )
        data = data.map(process_chunk)
        return data

    def create_dataset(self) -> EnvDataset:
        datasets = DatasetRegistry[EnvDataset]()
        pusht_datasets.register(datasets)
        lower_bounds.register_datasets(datasets)
        return datasets.create(self.dataset)
    
    @agt.jit
    def relative_action(self, obs, action):
        if self.dataset == "pusht/chi":
            from argon.envs.pusht import PushTAgentPos
            relative = action.agent_pos - obs.agent_pos
            return PushTAgentPos(relative)
        else:
            raise NotImplementedError()
    @agt.jit
    def absolute_action(self, obs, rel_action):
        if self.dataset == "pusht/chi":
            from argon.envs.pusht import PushTAgentPos
            absolute = rel_action.agent_pos + obs.agent_pos
            return PushTAgentPos(absolute)
        else:
            raise NotImplementedError()

    def load(self, dataset, splits=set()) -> tuple[Environment, dict[str, Data[Sample]]]:
        env = dataset.env(observation_type=self.observation_type)
        loaded_splits = {}
        if "train" in splits:
            logger.info(f"Loading training data from [blue]{self.dataset}[/blue]")
            train_data = dataset.split("train")
            if self.train_trajectories is not None:
                train_data = train_data.slice(0, self.train_trajectories)
            train_data = self._process_data(env, train_data)
            loaded_splits["train"] = train_data
        if "test" in splits:
            logger.info(f"Loading test data from [blue]{self.dataset}[/blue]")
            test_data = dataset.split("test")
            if self.test_trajectories is not None:
                test_data = test_data.slice(0, self.test_trajectories)
            test_data = self._process_data(env, test_data)
            loaded_splits["test"] = test_data
        if "validation" in splits:
            logger.info(f"Loading validation data from [blue]{self.dataset}[/blue]")
            validation_data = dataset.split("validation")
            if self.validation_trajectories is not None:
                validation_data = validation_data.slice(0, self.validation_trajectories)
            # get the first state of the trajectory
            validation_data = validation_data.truncate(1).map(
                lambda x: env.full_state(argon.tree.map(lambda y: y[0], x.reduced_state))
            )
            loaded_splits["validation"] = validation_data
        return env, loaded_splits

@struct(frozen=True, kw_only=True)
class Inputs:
    experiment: comet_ml.Experiment
    env_timesteps: int
    rng: PRNGSequence
    env: Environment
    dataset: EnvDataset

    validate : tp.Callable[[atyp.Array, Policy], tuple[Rollout, atyp.Array]]

    data : DataConfig

class Result:
    def create_policy(self) -> Policy:
        pass

class Method:
    def run(self, inputs: Inputs) -> Result:
        raise NotImplementedError()