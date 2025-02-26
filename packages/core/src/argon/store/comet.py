import comet_ml
import tempfile
import argon.store
import argon.numpy as npx
import argon.typing as atp
import numpy as np

from jax.tree_util import DictKey, GetAttrKey, SequenceKey

def to_artifact(*, name, artifact_type, data) -> comet_ml.Artifact:
    artifact = comet_ml.Artifact(name, artifact_type)
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/data.zarr"
        argon.store.dump(data, argon.store.LocalStore(zarr_path))
        artifact.add(zarr_path, "data.zarr", copy_to_tmp=False)
    return artifact

def _key_to_str(key):
    if isinstance(key, DictKey):
        return key.key
    elif isinstance(key, GetAttrKey):
        return key.name
    elif isinstance(key, SequenceKey):
        return str(key.index)
    else:
        return str(key)

def log(experiment : comet_ml.Experiment, iteration, *metrics, prefix=""):
    final_metrics = {}
    for m in metrics:
        leaves, _ = argon.tree.flatten_with_path(m)
        for k, v in leaves:
            if isinstance(v, (atp.Array, float, int, np.ndarray)):
                v = npx.mean(v).item()
                k = ".".join(_key_to_str(key) for key in  k)
                final_metrics[f"{prefix}{k}"] = v
    experiment.log_metrics(final_metrics, step=iteration)