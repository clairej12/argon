from argon.struct import struct
from argon.datasets.common import DatasetRegistry
from argon.datasets.envs.common import EnvDataset, Step
from argon.data import PyTreeData, idx_dtype
from argon.data.sequence import (
    SequenceInfo, SequenceData
)
from argon.datasets.util import download, cache_path
from argon.envs.pusht import (
    PushTEnv, PushTAgentPos,
    PositionControlTransform,
    PositionObsConfig, KeypointObsConfig,
    RelKeypointObsConfig, FullObsConfig
)
from argon.envs.mujoco import SystemState
from argon.envs.common import ChainedTransform, MultiStepTransform

import argon.store
import jax
import argon.numpy as jnp
import zarr
import zarr.storage

@struct(frozen=True)
class PushTDataset(EnvDataset[Step, PushTEnv]):
    _splits : dict[str, SequenceData[Step, None]]

    def split(self, name) -> SequenceData[Step, None]:
        return self._splits[name]

    def env(self, observation_type="positional", **kwargs):
        match observation_type:
            case "full" | None: obs_config = FullObsConfig()
            case "positional": obs_config = PositionObsConfig()
            case "keypoint": obs_config = KeypointObsConfig()
            case "rel_keypoint": obs_config = RelKeypointObsConfig()
            case _: raise ValueError(f"Unknown observation type: {observation_type}")
        env = PushTEnv(default_observation=obs_config)
        env = ChainedTransform([
            PositionControlTransform(),
            MultiStepTransform(10),
        ]).apply(env)
        return env

def load_pytorch_pusht_data(zarr_path, max_trajectories=None):
    store = zarr.storage.ZipStore(zarr_path, mode="r")
    meta = zarr.open(store, path="/meta", mode="r", zarr_format=2)
    data = zarr.open(store, path="/data", mode="r", zarr_format=2)
    if max_trajectories is None:
        max_trajectories = meta["episode_ends"].shape[0]
    ends = jnp.array(meta["episode_ends"][:max_trajectories], dtype=idx_dtype)
    starts = jnp.roll(ends, 1).at[0].set(0)
    last_end = ends[-1]
    infos = SequenceInfo(
        start_idx=starts,
        end_idx=ends,
        length=ends-starts,
        info=None
    )
    @jax.vmap
    def convert_states(state):
        agent_pos = jnp.array([1, -1], dtype=jnp.float32)*((state[:2] - 256) / 252)
        block_pos = jnp.array([1, -1], dtype=jnp.float32)*((state[2:4] - 256) / 252)
        block_rot = -state[4]
        # our rotation q is around the block center of mass
        # while theirs is around block_pos
        # we need to adjust the position accordingly
        # our_true_block_pos = our_block_body_q_pos + com_offset - our_q_rot @ com_offset
        # we substitute our_true_pos for block_pos and solve
        rotM = jnp.array([
            [jnp.cos(block_rot), -jnp.sin(block_rot)],
            [jnp.sin(block_rot), jnp.cos(block_rot)]
        ], dtype=jnp.float32)
        block_scale = 30/252
        com = 0.5*(block_scale/2) + 0.5*(2.5*block_scale)
        com = jnp.array([0, -com], dtype=jnp.float32)
        block_pos = block_pos + rotM @ com - com

        q = jnp.concatenate([agent_pos, block_pos, block_rot[None]])
        return SystemState(
            time=jnp.ones((), dtype=jnp.float32), 
            qpos=q, qvel=jnp.zeros_like(q),
            act=jnp.zeros((0,), dtype=jnp.float32)
        )
    @jax.vmap
    def convert_actions(action):
        return PushTAgentPos(action / 256 - 1)

    states = convert_states(jnp.array(data["state"][:last_end], dtype=jnp.float32))
    # actions = convert_actions(jnp.array(data["action"][:last_end], dtype=jnp.float32))
    # get the next block position and use that for the action
    positions = states.qpos[:, :2]
    # make the action 2 states ahead
    actions = jnp.concatenate([positions[1:], positions[-1][None]])
    # set the last action to the last position for each the sequence
    actions.at[infos.end_idx-1].set(positions[infos.end_idx - 1])
    actions = PushTAgentPos(actions)

    steps = Step(
        state=None,
        reduced_state=states,
        observation=None,
        action=actions
    )
    return SequenceData(PyTreeData(steps), PyTreeData(infos))

def load_chi_pusht_data(max_trajectories=None, quiet=False):
    zip_path = cache_path("pusht", "pusht_data_raw.zarr.zip")
    processed_path = cache_path("pusht", "pusht_data.zarr.zip")
    if not processed_path.exists():
        download(zip_path,
            job_name="PushT (Diffusion Policy Data)",
            gdrive_id="1ALI_Ua7U1EJRCAim5tvtQUJbBP5MGMyR",
            md5="48a64828d7f2e1e8902a97b57ebd0bdd",
            quiet=quiet
        )
        data = load_pytorch_pusht_data(zip_path, max_trajectories)
        argon.store.dump(data, processed_path)
        # remove the raw data
        zip_path.unlink()
    data = argon.store.load(processed_path)
    return data

def load_chi_pusht(quiet=False, train_trajs=None, test_trajs=10):
    data = load_chi_pusht_data()
    train = data.slice(0, len(data) - 32)
    validation = data.slice(len(data) - 32, 16)
    test = data.slice(len(data) - 16, 16)
    return PushTDataset(
        _splits={"train": train, "validation": validation, "test": test},
    )

def register(registry: DatasetRegistry, prefix=None):
    registry.register("pusht/chi", load_chi_pusht, prefix=prefix)