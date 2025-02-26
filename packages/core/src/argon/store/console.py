import logging
_logger = logging.getLogger("argon.train")

import numpy as np
import argon.numpy as npx
import argon.tree
import argon.typing as atp

from jax.tree_util import DictKey, GetAttrKey, SequenceKey

def _key_to_str(key):
    if isinstance(key, DictKey):
        return key.key
    elif isinstance(key, GetAttrKey):
        return key.name
    elif isinstance(key, SequenceKey):
        return str(key.index)
    else:
        return str(key)

def log(iteration, *metrics, logger=None, prefix=""):
    logger = logger or _logger
    for m in metrics:
        leaves, _ = argon.tree.flatten_with_path(m)
        for k, v in leaves:
            if isinstance(v, (atp.Array, float, int, np.ndarray)):
                v = npx.mean(v).item()
                k = ".".join(_key_to_str(key) for key in  k)
                if iteration is not None:
                    logger.info(f"{iteration}: {prefix}{k}: {v}")
                else:
                    logger.info(f"{prefix}{k}: {v}")