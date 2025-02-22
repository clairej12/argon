import argon.typing as atp
import argon.transforms as agt

from argon.envs.common import Environment
from argon.struct import struct, field


import typing as tp
import mujoco
import mujoco.mjx as mjx

from .common import Action, Observation

@struct(frozen=True)
class SystemState:
    time: atp.Array
    qpos: atp.Array
    qvel: atp.Array
    act: atp.Array # actuator state

MujocoState = mjx.Data

@struct(frozen=True)
class MujocoEnvironment(tp.Generic[Action, Observation], Environment[MujocoState, SystemState, Action, Observation]):
    mjx_model: mjx.Model = field(init=False)

    def __post_init__(self):
        mjx_model = mjx.put_model(self.load_model())
        object.__setattr__(self, "mjx_model", mjx_model)

    def load_model(self) -> mujoco.MjModel:
        raise NotImplementedError()

    @agt.jit
    def full_state(self, reduced_state: SystemState) -> MujocoState:
        data = mjx.make_data(self.mjx_model)
        data = data.replace(
            time=reduced_state.time,
            qpos=reduced_state.qpos,
            qvel=reduced_state.qvel,
            act=reduced_state.act
        )
        return mjx.forward(self.mjx_model, data)
    
    @agt.jit
    def reduce_state(self, full_state):
        return SystemState(
            time=full_state.time,
            qpos=full_state.qpos,
            qvel=full_state.qvel,
            act=full_state.act
        )
    
    @agt.jit
    def sample_state(self, rng_key):
        return self.full_state(SystemState(
            time=atp.zeros((), atp.float32),
            qpos=atp.zeros((self.mjx_model.nq,), atp.float32),
            qvel=atp.zeros((self.mjx_model.nv,), atp.float32),
            act=atp.zeros((self.mjx_model.na,), atp.float32)
        ))

    @agt.jit
    def step(self, state: MujocoState, action: atp.Array | None, rng_key: atp.PRNGKey | None = None):
        if action is not None:
            state = state.replace(ctrl=action)
        return mjx.step(self.mjx_model, state)