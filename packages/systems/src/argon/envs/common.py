import typing
import abc

import argon.transforms as agt
import argon.typing as atyp
import argon.transforms as agt
import argon.random

from argon.struct import struct
from argon.registry import Registry
from typing import Optional, Any

State = typing.TypeVar("State")
ReducedState = typing.TypeVar("ReducedState")
Action = typing.TypeVar("Action")
Observation = typing.TypeVar("Observation")

class ObservationConfig(typing.Generic[Observation]): ...

class Environment(typing.Generic[State, ReducedState, Action, Observation], abc.ABC):
    # By default ReducedState = FullState
    # The notion of "full" and "reduced states" are useful for
    # environments that have cached information in the state.
    # (active contacts, contact forces, etc.)
    # As a result we may only approximately have:
    # step(full_state(reduced_state(s))) ~= step(s)
    # up to some solver error tolerance.

    def full_state(self, reduced_state: ReducedState) -> State:
        return reduced_state

    def reduce_state(self, full_state: State) -> ReducedState:
        return full_state

    @abc.abstractmethod
    def sample_state(self, rng_key : atyp.Array) -> State: ...
    @abc.abstractmethod
    def sample_action(self, rng_key : atyp.Array) -> Action: ...
    @abc.abstractmethod
    def reset(self, rng_key : atyp.Array) -> State: ...
    @abc.abstractmethod
    def step(self, state : State, action : Action,
             rng_key : Optional[atyp.Array] = None) -> State: ...
    @abc.abstractmethod
    def is_finished(self, state: State) -> atyp.Array: ...
    @abc.abstractmethod
    def observe(self, state: State,
            config: ObservationConfig[Observation] | None = None) -> Observation: ...
    @abc.abstractmethod
    def reward(self, state: State,
               action : Action, next_state : State) -> atyp.Array: ...
    @abc.abstractmethod
    def trajectory_reward(self, states: State, actions : Action) -> atyp.Array: ...
    @abc.abstractmethod
    def cost(self, states: State, actions: Action) -> atyp.Array: ...

    @abc.abstractmethod
    def visualize(self, states: State, actions: Action | None = None, **kwargs) -> Any: ...

@struct(frozen=True)
class EnvWrapper(Environment[State, ReducedState, Action, Observation]):
    base: Environment[State, ReducedState, Action, Observation]

    # We need to manually override these since
    # they are already defined in Environment
    # and therefore not subject to __getattr__
    def full_state(self, reduced_state: ReducedState) -> State:
        return self.base.full_state(reduced_state)
    def reduce_state(self, full_state: State) -> ReducedState:
        return self.base.reduce_state(full_state)
    def sample_state(self, rng_key : atyp.Array) -> State:
        return self.base.sample_state(rng_key)
    def sample_action(self, rng_key : atyp.Array) -> Action:
        return self.base.sample_action(rng_key)
    def reset(self, rng_key : atyp.Array) -> State:
        return self.base.reset(rng_key)
    def step(self, state : State, action : Action,
             rng_key : Optional[atyp.Array] = None) -> State:
        return self.base.step(state, action, rng_key)
    def observe(self, state: State, config: ObservationConfig[Observation] | None = None) -> Observation:
        return self.base.observe(state, config)
    def reward(self, state: State,
               action : Action, next_state : State) -> atyp.Array:
        return self.base.reward(state, action, next_state)
    def trajectory_reward(self, states: State, actions : Action) -> atyp.Array:
        return self.base.trajectory_reward(states, actions)
    def cost(self, states: State, actions: Action) -> atyp.Array:
        return self.base.cost(states, actions)
    def is_finished(self, state: State) -> atyp.Array:
        return self.base.is_finished(state)
    def visualize(self, states: State, actions: Action | None = None, **kwargs) -> Any:
        return self.base.visualize(states, actions, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)

class EnvTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, env : Environment) -> Environment: ...

@struct(frozen=True)
class ChainedTransform(EnvTransform):
    transforms: list[EnvTransform]

    def apply(self, env):
        for t in self.transforms:
            env = t.apply(env)
        return env

@struct(frozen=True)
class MultiStepTransform(EnvTransform):
    steps: int = 1

    def apply(self, env):
        return MultiStepEnv(env, self.steps)

@struct(frozen=True)
class MultiStepEnv(EnvWrapper):
    steps: int = 1

    @agt.jit
    def step(self, state, action, rng_key=None):
        keys = argon.random.split(rng_key, self.steps) \
            if rng_key is not None else None
        def step_fn(state, key):
            state = self.base.step(state, action, key)
            return state, None
        state, _ = agt.scan(step_fn, length=self.steps)(state, keys)
        return state

@struct(frozen=True)
class ImageRender(ObservationConfig[atyp.Array]):
    width: int = 256
    height: int = 256
    camera: int | str | None = None

@struct(frozen=True)
class ImageActionsRender(ImageRender):
    actions: typing.Any = None

EnvironmentRegistry = Registry

class EnvTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, env): ...

@struct(frozen=True)
class ChainedTransform(EnvTransform):
    transforms: list[EnvTransform]

    def apply(self, env):
        for t in self.transforms:
            env = t.apply(env)
        return env

@struct(frozen=True)
class MultiStepTransform(EnvTransform):
    steps: int = 1

    def apply(self, env):
        return MultiStepEnv(env, self.steps)


@struct(frozen=True)
class MultiStepEnv(EnvWrapper):
    steps: int = 1

    @agt.jit
    def step(self, state, action, rng_key=None):
        keys = argon.random.split(rng_key, self.steps) \
            if rng_key is not None else None
        def step_fn(state, key):
            state = self.base.step(state, action, key)
            return state, None
        state, _ = agt.scan(step_fn, length=self.steps)(state, keys)
        return state

    def visualize(self, states, actions = None, *, 
                    type: str = "html", dt=None, **kwargs) -> str:
        if dt is None: dt = 1
        return self.base.visualize(states, actions,
            type=type,
            dt=self.steps*dt,
            **kwargs
        )