import argon.transforms as agt
import argon.numpy as npx
import argon.typing as atyp
import argon.tree

from argon.random import PRNGSequence
from argon.envs.common import Environment, ObservationConfig

from argon.datasets.common import DatasetRegistry
from argon.datasets.envs.common import EnvDataset, Step
from argon.datasets.envs import pusht as pusht_datasets

from argon.struct import struct
from argon.policy import Policy
from argon.data import Data

import typing as tp
from ml_collections import ConfigDict

import logging

logger = logging.getLogger(__name__)

@struct(frozen=True)
class Sample:
    state: tp.Any
    observations: atyp.Array
    actions: atyp.Array

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
        cd.observation_type = None
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
            states, actions = chunk.elements
            actions = argon.tree.map(lambda x: x[-self.action_length:], actions)
            obs_states = argon.tree.map(lambda x: x[:self.obs_length], states)
            curr_state = argon.tree.map(lambda x: x[-1], obs_states)
            obs = agt.vmap(env.observe)(obs_states)
            return Sample(
                curr_state, obs, actions
            )
        data = data.map(process_chunk)
        return data
    
    def create_dataset(self) -> EnvDataset:
        datasets = DatasetRegistry[EnvDataset]()
        pusht_datasets.register(datasets)
        return datasets.create(self.dataset)
    
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
    env_timesteps: int
    rng: PRNGSequence
    env: Environment
    dataset: EnvDataset

    validate : tp.Callable[[atyp.Array, Policy], atyp.Array]

    data : DataConfig

class Result:
    def create_policy(self) -> Policy:
        pass

class Method:
    def run(self, inputs: Inputs) -> Result:
        raise NotImplementedError()