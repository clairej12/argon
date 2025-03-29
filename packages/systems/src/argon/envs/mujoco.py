import argon.typing as atp
import argon.transforms as agt
import argon.tree

from argon.envs.common import Environment
from argon.struct import struct, field


import importlib.resources as resources
import typing as tp
import mujoco
import mujoco.mjx as mjx

import json
import zlib
import jax
import numpy as np
import base64

from .common import Action, Observation
from . import assets

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

    def visualize(self, states: MujocoState, actions: tp.Any = None, *, 
                    type: str = "html", dt=None,
                    extras_geoms: dict = {}, **kwargs) -> str:
        if type != "html":
            raise ValueError()
        if dt is None: dt = 1
        dt = float(dt)*self.mjx_model.opt.timestep.item()
        extra_dims = states.xpos.ndim - 2
        xpos, xquat = states.xpos, states.xquat
        if extra_dims == 2:
            # For now take the first trajectory
            xpos, xquat = xpos[0], xquat[0]
        elif extra_dims == 0:
            xpos, xquat = xpos[None], xquat[None]
        elif extra_dims != 1:
            raise ValueError("Illegal number of extra dimensions")
        json_str = _dump_json(
            self.mjx_model, xpos, xquat, dt=dt,
            extra_geoms=extras_geoms
        )
        json_str = base64.b64encode(
            zlib.compress(json_str.encode('utf-8'))
        ).decode('ascii')
        js_url = kwargs.pop("js_url", None)
        if js_url is None:
            base_url = 'https://cdn.jsdelivr.net/gh/google/brax'
            js_url = f'{base_url}@v0.12.1/brax/visualizer/js/viewer.js'
        deps = dict(_DEPENDENCIES)
        deps["viewer"] = js_url
        pako_src = deps["pako"]
        del deps["pako"]
        import_map = ",\n".join(f'"{k}": "{v}"' for k, v in deps.items())
        with (resources.files(assets) / "visualizer.html").open("r") as f:
            html = f.read()
            html = html.replace("{{ import_map }}", import_map)
            html = html.replace("{{ pako_src }}", pako_src)
            html = html.replace("{{ system_json_b64 }}", json_str)
        html = f"""
        <iframe srcdoc="{html.replace('"', '&quot;')}" style="width: 100%; height:450px; border: none"></iframe>
        """
        return Html(html)

class Html(str):
    def _repr_html_(self):
        return str(self)

_DEPENDENCIES = {
    "pako": "https://unpkg.com/pako@2.1.0/dist/pako.min.js",
    "three": "https://unpkg.com/three@0.150.1/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.150.1/examples/jsm/",
    "lilgui": "https://cdn.jsdelivr.net/npm/lil-gui@0.18.0/+esm",
}

def _dump_json(model: mjx.Model, xpos: atp.Array, xquat: atp.Array,
               dt=None, extra_geoms={}) -> str:
    d = _to_dict(model)
    # fill in empty link names
    link_names = [
        _addr_to_str(model.names, idx) or f'link {i}' 
        for i, idx in enumerate(model.name_bodyadr[1:])
    ]
    for name in extra_geoms:
        if name not in link_names:
            link_names.append(name)
    link_names.append("world")
    d["link_names"] = link_names
    if dt is not None:
        d["opt"]["timestep"] = dt
    # unpack geoms into a dict for the visualizer
    link_geoms = {}
    for id_ in range(model.ngeom):
        link_idx = model.geom_bodyid[id_] - 1
        rgba = np.array(model.geom_rgba[id_])
        if (rgba == [0.5, 0.5, 0.5, 1.0]).all():
            # convert the default mjcf color to our default color
            rgba = np.array([0.4, 0.33, 0.26, 1.0])
        geom = {
            'name': _GEOM_TYPE_NAMES[model.geom_type[id_]],
            'link_idx': link_idx,
            'pos': model.geom_pos[id_],
            'rot': model.geom_quat[id_],
            'rgba': rgba,
            'size': model.geom_size[id_],
        }
        if geom['name'] == 'Mesh':
            vert, face = _get_mesh(model, model.geom_dataid[id_])
            geom['vert'] = vert
            geom['face'] = face
        link_geoms.setdefault(link_names[link_idx], []).append(_to_dict(geom))
    link_names.pop() # remove world
    for name, geoms in extra_geoms.items():
        link_idx = link_names.index(name)
        lgs = link_geoms.setdefault(name, [])
        for value in geoms:
            value = dict(value)
            value['link_idx'] = link_idx
            if 'rgba' not in value:
                rgba = np.array([0.4, 0.33, 0.26, 1.0])
            lgs.append(_to_dict(value))

    d['geoms'] = link_geoms
    # add states for the viewer, we only need 'x' (positions and orientations).
    assert xpos.shape[0] == xquat.shape[0]
    assert xpos.ndim == xquat.ndim == 3
    # Don't include world in the states
    d['states'] = {'x': [
        {'pos': _to_dict(xpos[i,1:]), 'rot': _to_dict(xquat[i,1:])}
        for i in range(xpos.shape[0])
    ]}
    return json.dumps(d)

def _addr_to_str(buf: bytes, pos):
   end = buf.find(b'\0', pos + 1)
   if end != -1: return buf[pos:end].decode()
   else: return buf[pos:]


def _to_dict(obj):
    """Converts python object to a json encodeable object."""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [_to_dict(s) for s in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items() if k in _ENCODE_FIELDS}
    if isinstance(obj, jax.Array):
        return _to_dict(obj.tolist())
    if hasattr(obj, '__dict__'):
        d = dict(obj.__dict__)
        d['name'] = obj.__class__.__name__
        return _to_dict(d)
    if isinstance(obj, np.ndarray):
        return _to_dict(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    return obj

def _get_mesh(mj: mujoco.MjModel, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Gets mesh from mj at index i."""
    last = (i + 1) >= mj.nmesh
    face_start = mj.mesh_faceadr[i]
    face_end = mj.mesh_faceadr[i + 1] if not last else mj.mesh_face.shape[0]
    face = mj.mesh_face[face_start:face_end]
    vert_start = mj.mesh_vertadr[i]
    vert_end = mj.mesh_vertadr[i + 1] if not last else mj.mesh_vert.shape[0]
    vert = mj.mesh_vert[vert_start:vert_end]
    return vert, face

_ENCODE_FIELDS = [
    'opt',
    'timestep',
    'face',
    'size',
    'link_idx',
    'link_names',
    'name',
    'dist',
    'pos',
    'rgba',
    'rot',
    'states',
    'transform',
    'vert',
    'x',
    'xd',
]

_GEOM_TYPE_NAMES = {
    0: 'Plane',
    1: 'HeightMap',
    2: 'Sphere',
    3: 'Capsule',
    5: 'Cylinder',
    6: 'Box',
    7: 'Mesh',
}