import logging
_logger = logging.getLogger("argon.train")

import numpy as np
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

def log(iteration, *metrics, logger=None):
    logger = logger or _logger
    for m in metrics:
        leaves, _ = argon.tree.flatten_with_path(m)
        for k, v in leaves:
            if isinstance(v, (atp.Array, float, int, np.ndarray)):
                k = ".".join(_key_to_str(key) for key in  k)
                logger.info(f"{iteration}: {k}: {v}")