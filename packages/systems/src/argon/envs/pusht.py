import argon.numpy as npx
import argon.random
import argon.typing as atp
import argon.transforms as agt
import argon.tree

from argon.struct import struct, field

from .common import ObservationConfig, EnvWrapper, EnvTransform
from .mujoco import MujocoEnvironment, MujocoState, SystemState
from . import assets, util

import argon.typing as atp

import importlib.resources as resources
import mujoco
import shapely.geometry as sg


@struct(frozen=True)
class PushTObs:
    agent_pos: atp.Array = None
    agent_vel: atp.Array = None

    block_pos: atp.Array = None
    block_vel: atp.Array = None

    block_rot: atp.Array = None
    block_rot_vel: atp.Array = None

@struct(frozen=True)
class PushTAction:
    agent_force: atp.Array

@struct(frozen=True)
class PushTAgentPos:
    agent_pos: atp.Array

@struct(frozen=True)
class PushTPosObs:
    agent_pos: atp.Array
    block_pos: atp.Array
    block_rot: atp.Array

@struct(frozen=True)
class PushTKeypointObs:
    agent_pos: atp.Array
    block_pos: atp.Array
    block_end: atp.Array

@struct(frozen=True)
class PushTKeypointRelObs:
    agent_block_pos: atp.Array
    agent_block_end: atp.Array
    rel_block_pos: atp.Array
    rel_block_end: atp.Array

@struct(frozen=True)
class FullObsConfig(ObservationConfig[PushTPosObs]): ...
@struct(frozen=True)
class PositionObsConfig(ObservationConfig[PushTPosObs]): ...
@struct(frozen=True)
class KeypointObsConfig(ObservationConfig[PushTPosObs]): ...
@struct(frozen=True)
class RelKeypointObsConfig(ObservationConfig[PushTPosObs]): ...

@struct(frozen=True)
class PushTEnv(MujocoEnvironment):
    goal_pos: atp.Array = field(default_factory=lambda: npx.zeros((2,), npx.float32))
    goal_rot: atp.Array = field(default_factory=lambda: npx.array(-npx.pi/4, npx.float32))
    default_observation : ObservationConfig = PositionObsConfig()

    success_threshold: float = 0.85

    agent_radius : float = 15 / 252
    block_scale : float = 30 /252
    world_scale : float = 2

    def load_xml(self):
        with (resources.files(assets) / "pusht.xml").open("r") as f:
            xml = f.read()
        com = 0.5*(self.block_scale/2) + 0.5*(self.block_scale + 1.5*self.block_scale)
        return xml.format(
            agent_radius=self.agent_radius,
            world_scale=self.world_scale,
            half_world_scale=self.world_scale/2,
            # the other constants needed for the block
            block_scale=self.block_scale,
            half_block_scale=self.block_scale/2,
            double_block_scale=2*self.block_scale,
            one_and_half_block_scale=1.5*self.block_scale,
            two_and_half_block_scale=2.5*self.block_scale,
            com_offset=com
        )
    
    def load_model(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(self.load_xml())

    @agt.jit
    def reset(self, rng_key : atp.PRNGKey) -> MujocoState:
        a_pos, b_pos, b_rot, c = argon.random.split(rng_key, 4)
        agent_pos = argon.random.uniform(a_pos, (2,), minval=-0.8, maxval=0.8)
        block_rot = argon.random.uniform(b_pos, (), minval=-npx.pi, maxval=npx.pi)
        block_pos = argon.random.uniform(b_rot, (2,), minval=-0.4, maxval=0.4)
        # re-generate block positions while the block is too close to the agent
        min_radius = self.block_scale*2*npx.sqrt(2) + self.agent_radius
        def gen_pos(carry):
            rng_key, _ = carry
            rng_key, sk = argon.random.split(rng_key)
            return (rng_key, argon.random.uniform(sk, (2,), minval=-0.4, maxval=0.4))
        _, block_pos = agt.while_loop(
            lambda s: npx.linalg.norm(s[1] - agent_pos) < min_radius,
            gen_pos, (c, block_pos)
        )
        qpos = npx.concatenate([agent_pos, block_pos, block_rot[npx.newaxis]])
        return self.full_state(SystemState(
            npx.zeros((), dtype=npx.float32), 
            qpos, 
            npx.zeros_like(qpos), 
            npx.zeros((0,), dtype=npx.float32)
        ))

    @agt.jit
    def sample_state(self, rng_key):
        return self.reset(rng_key)
    
    @agt.jit
    def sample_action(self, rng_key):
        return PushTAction(argon.random.normal(rng_key, (2,)))
    
    @agt.jit
    def step(self, state: MujocoState, action: PushTAction, rng_key : atp.PRNGKey | None = None):
        return super().step(state, action.agent_force)

    @agt.jit
    def is_finished(self, state: MujocoState):
        return self.reward(None, None, state) >= 1.

    @agt.jit
    def observe(self, state : MujocoState, config : ObservationConfig | None = None):
        if config is None: config = self.default_observation
        obs = PushTObs(
            # Extract agent pos, vel
            agent_pos=state.xpos[1,:2],
            agent_vel=state.cvel[1,3:5],
            # Extract block pos, vel, angle, angular vel
            block_pos=state.xpos[2,:2],
            block_rot=util.quat_to_angle(state.xquat[2,:4]),
            block_vel=state.cvel[2,3:5],
            block_rot_vel=state.cvel[2,2],
        )
        if isinstance(config, FullObsConfig):
            return obs
        elif isinstance(config, PositionObsConfig):
            return PushTPosObs(
                agent_pos=obs.agent_pos,
                block_pos=obs.block_pos,
                block_rot=obs.block_rot
            )
        elif isinstance(config, KeypointObsConfig):
            end = util.angle_to_rot2d(obs.block_rot) @ npx.array([0, -4*self.block_scale])
            return PushTKeypointObs(
                agent_pos=obs.agent_pos,
                block_pos=obs.block_pos,
                block_end=end
            )
        elif isinstance(config, RelKeypointObsConfig):
            end = util.angle_to_rot2d(obs.block_rot) @ npx.array([0, -4*self.block_scale])
            goal_end = util.angle_to_rot2d(self.goal_rot) @ npx.array([0, -4*self.block_scale])
            return PushTKeypointRelObs(
                agent_block_pos=obs.agent_pos - obs.block_pos,
                agent_block_end=obs.agent_pos - end,
                rel_block_pos=obs.block_pos - self.goal_pos,
                rel_block_end=end - goal_end,
            )


    # For computing the reward
    def _block_points(self, pos, rot):
        center_a, hs_a = npx.array([0, -self.block_scale/2], dtype=npx.float32), \
                npx.array([2*self.block_scale, self.block_scale/2], dtype=npx.float32)
        center_b, hs_b = npx.array([0, -2.5*self.block_scale], dtype=npx.float32), \
                        npx.array([self.block_scale/2, 1.5*self.block_scale], dtype=npx.float32)

        points = npx.array([
            center_a + npx.array([hs_a[0], -hs_a[1]], dtype=npx.float32),
            center_a + hs_a,
            center_a + npx.array([-hs_a[0], hs_a[1]], dtype=npx.float32),
            center_a - hs_a,
            center_b + npx.array([-hs_b[0], hs_b[1]], dtype=npx.float32),
            center_b - hs_b,
            center_b + npx.array([hs_b[0], -hs_b[1]], dtype=npx.float32),
            center_b + hs_b
        ])
        rotM = npx.array([
            [npx.cos(rot), -npx.sin(rot)],
            [npx.sin(rot), npx.cos(rot)]
        ], dtype=npx.float32)
        points = agt.vmap(lambda v: rotM @ v)(points)
        return points + pos

    @agt.jit
    def reward(self, state : MujocoState, 
                action : PushTAction, 
                next_state : MujocoState):
        obs = self.observe(next_state)
        goal_points = self._block_points(self.goal_pos, self.goal_rot)
        points = self._block_points(obs.block_pos, obs.block_rot)
        overlap = util.polygon_overlap(goal_points, points)
        return npx.minimum(overlap, self.success_threshold) / self.success_threshold

    @agt.jit
    def trajectory_reward(self, states: MujocoState, actions: PushTAction):
        prev_states = argon.tree.map(lambda x: x[:-1], states)
        next_states = argon.tree.map(lambda x: x[1:], states)
        if argon.tree.axis_size(states, 0) == argon.tree.axis_size(actions, 0):
            actions = argon.tree.map(lambda x: x[:-1], actions)
        rewards = agt.vmap(self.reward)(prev_states, actions, next_states)
        # use the max reward over the trajectory
        return npx.max(rewards)
    
    @agt.jit
    def cost(self, states, actions):
        return -self.trajectory_reward(states, actions)

        
# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@struct(frozen=True)
class PositionControlTransform(EnvTransform):
    k_p : float = 15
    k_v : float = 2
    
    def apply(self, env):
        return PositionControlEnv(env, self.k_p, self.k_v)

@struct(frozen=True)
class PositionControlEnv(EnvWrapper):
    k_p : float = 50
    k_v : float = 2

    def sample_action(self, rng_key):
        return super().sample_action(rng_key)

    def step(self, state, action : PushTAgentPos, rng_key=None):
        obs = self.base.observe(state, config=FullObsConfig())
        if action is not None:
            a = self.k_p * (action.agent_pos - obs.agent_pos) + self.k_v * (-obs.agent_vel)
        else: 
            a = npx.zeros((2,), dtype=npx.float32)
        a = PushTAction(a)
        return self.base.step(state, a, None)

def register_all(registry, prefix=None):
    registry.register("pusht", PushTEnv, prefix=prefix)