import logging
import rich
import argparse
import sys
import typing as tp
import jax
import os

from pathlib import Path
from ml_collections import ConfigDict
from rich.logging import RichHandler

FORMAT = "%(name)s - %(message)s"

class Missing:
    def __repr__(self):
        return "???"
MISSING = Missing()

class CustomLogRender(rich._log_render.LogRender):
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

def setup_cache():
    cache_dir = Path(os.environ.get("HOME")) / ".cache" / "jax_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.25)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")



def setup_logging(show_path=False):
    jax_logger = logging.getLogger("jax")
    if len(jax_logger.handlers) > 0:
        jax_logger.removeHandler(jax_logger.handlers[0])
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.WARNING)
    logging.getLogger("argon").setLevel(logging.INFO)

    console = rich.get_console()
    handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_path=show_path,
        console=console
    )
    renderer = CustomLogRender(
        show_time=handler._log_render.show_time,
        show_level=handler._log_render.show_level,
        show_path=handler._log_render.show_path,
        time_format=handler._log_render.time_format,
        omit_repeated_times=handler._log_render.omit_repeated_times,
    )
    handler._log_render = renderer
    logging.basicConfig(
        level=logging.ERROR,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[handler]
    )

def _is_float(v):
    try:
        float(v)
        return True
    except:
        return False

def parse_options(cd: ConfigDict, args: tp.Iterable[str] | None = None):
    if args is None: args = sys.argv[1:]
    def _options(cd: ConfigDict, prefix=""):
        for k, v in cd.items():
            if isinstance(v, ConfigDict):
                yield from _options(v, f"{prefix}{k}.")
            else:
                yield (f"{prefix}{k}", cd, k)
    options = list(_options(cd))
    parser = argparse.ArgumentParser()
    for opt, _, _ in options:
        parser.add_argument(f"--{opt}", type=str, required=False, default=MISSING)
    args = parser.parse_args(args)
    for opt, cd, k in options:
        v = getattr(args, opt)
        if v is MISSING:
            continue
        tp = type(cd[k]) if cd[k] is not None else None
        if tp is None:
            if v.lower() == "true" or v.lower() == "false":
                tp = bool
            if v.isdecimal(): tp = int
            elif _is_float(v): tp = float
            else: tp = str
        if tp == bool:
            v = v.lower() == "true"
        else:
            v = tp(v)
        cd[k] = v
