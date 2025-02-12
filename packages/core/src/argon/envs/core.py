import typing
import abc

import argon.core as F
from argon.core.dataclasses import dataclass
from argon.util.registry import Registry
from typing import Optional

State = typing.TypeVar("State")
ReducedState = typing.TypeVar("ReducedState")
Action = typing.TypeVar("Action")
Observation = typing.TypeVar("Observation")
Render = typing.TypeVar("Render")

class RenderConfig(typing.Generic[Render]): ...
class ObserveConfig(typing.Generic[Observation]): ...

class Environment(typing.Generic[State, ReducedState, Action]):
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

    def sample_state(self, rng_key : F.Array) -> State:
        raise NotImplementedError()

    def sample_action(self, rng_key : F.Array) -> Action:
        raise NotImplementedError()

    def reset(self, rng_key : F.Array) -> State:
        raise NotImplementedError()

    def step(self, state : State, action : Action,
             rng_key : Optional[F.Array] = None) -> State:
        raise NotImplementedError()
            
    def observe(self, state: State,
            config: ObserveConfig[Observation] | None = None) -> Observation:
        raise NotImplementedError()

    def reward(self, state: State,
               action : Action, next_state : State) -> F.Array:
        raise NotImplementedError()

    def total_reward(self, states: State, actions : Action) -> F.Array:
        raise NotImplementedError()

    def cost(self, states: State, actions: Action) -> F.Array:
        raise NotImplementedError()

    def is_finished(self, state: State) -> F.Array:
        raise NotImplementedError()

    def render(self, state : State,
               config: RenderConfig[Render] | None = None) -> Render:
        raise NotImplementedError()

@dataclass
class EnvWrapper(Environment[State, ReducedState, Action]):
    base: Environment[State, ReducedState, Action]

    # We need to manually override these since
    # they are already defined in Environment
    # and therefore not subject to __getattr__

    def full_state(self, reduced_state: ReducedState) -> State:
        return self.base.full_state(reduced_state)
    def reduce_state(self, full_state: State) -> ReducedState:
        return self.base.reduce_state(full_state)
    def sample_state(self, rng_key : F.Array) -> State:
        return self.base.sample_state(rng_key)
    def sample_action(self, rng_key : F.Array) -> Action:
        return self.base.sample_action(rng_key)
    def reset(self, rng_key : F.Array) -> State:
        return self.base.reset(rng_key)
    def step(self, state : State, action : Action,
             rng_key : Optional[F.Array] = None) -> State:
        return self.base.step(state, action, rng_key)
    def observe(self, state: State,
            config: ObserveConfig[Observation] | None = None) -> Observation:
        return self.base.observe(state, config)
    def reward(self, state: State,
               action : Action, next_state : State) -> F.Array:
        return self.base.reward(state, action, next_state)
    def cost(self, states: State, actions: Action) -> F.Array:
        return self.base.cost(states, actions)
    def is_finished(self, state: State) -> F.Array:
        return self.base.is_finished(state)
    def render(self, state : State,
               config: RenderConfig[Render] | None = None) -> Render:
        return self.base.render(state, config)

    def __getattr__(self, name):
        return getattr(self.base, name)

class EnvTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, env : Environment) -> Environment: ...

@dataclass
class ChainedTransform(EnvTransform):
    transforms: list[EnvTransform]

    def apply(self, env):
        for t in self.transforms:
            env = t.apply(env)
        return env

@dataclass
class MultiStepTransform(EnvTransform):
    steps: int = 1

    def apply(self, env):
        return MultiStepEnv(env, self.steps)

@dataclass
class MultiStepEnv(EnvWrapper):
    steps: int = 1

    @agt.jit
    def step(self, state, action, rng_key=None):
        keys = argon.random.split(rng_key, self.steps) \
            if rng_key is not None else None
        def step_fn(state, key):
            state = self.base.step(state, action, key)
            return state, None
        state, _ = agt.scan(step_fn, state, keys, length=self.steps)
        return state

@dataclass
class ImageRender(RenderConfig[F.Array]):
    width: int = 256
    height: int = 256
    camera: int | str | None = None

@dataclass
class ImageActionsRender(ImageRender):
    actions: typing.Any = None

EnvironmentRegistry = Registry