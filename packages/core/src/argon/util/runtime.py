import os
import sys
import rich
import rich.jupyter
import rich.logging
import rich.terminal_theme
import rich._log_render
import subprocess
import multiprocessing
import argparse
import abc
import functools
import typing

from typing import Literal, Sequence, Callable, Type
from rich.text import Text
from rich.logging import RichHandler
from pathlib import Path
from bdb import BdbQuit

import logging
logger = logging.getLogger("argon")

# Make not expand
class CustomLogRender(rich._log_render.LogRender):
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

LOGGING_SETUP = False
def setup_logger(show_path=True):
    global LOGGING_SETUP
    FORMAT = "%(name)s - %(message)s"
    if not LOGGING_SETUP:
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
        LOGGING_SETUP = True
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("argon")
    logger.setLevel(logging.DEBUG)
    # Prevent "retrying" warnings from connectionpool
    # if running wandb offline
    logger = logging.getLogger("urllib3.connectionpool")
    logger.setLevel(logging.ERROR)

SETUP_JAX_CACHE = False

def setup_jax_cache():
    global SETUP_JAX_CACHE
    if SETUP_JAX_CACHE:
        return
    from jax.experimental.compilation_cache import compilation_cache as cc
    import tempfile
    user = os.environ.get("USER", "argon")
    JAX_CACHE = Path(tempfile.gettempdir()) / f"jax_cache_{user}"
    JAX_CACHE = Path(os.environ.get("JAX_CACHE", JAX_CACHE))
    JAX_CACHE.mkdir(parents=True, exist_ok=True)
    cc.initialize_cache(str(JAX_CACHE))
    SETUP_JAX_CACHE = True

SETUP_GC = False

def _rich_live_refresh(self):
    with self._lock:
        self._live_render.set_renderable(self.renderable)
        from IPython.display import TextDisplayObject
        def _render_to_text():
            loopback = io.StringIO()
            self.loopback_console.file = loopback
            with self.loopback_console:
                self.loopback_console.print(self._live_render.renderable)
            value = loopback.getvalue()
            return value
        self.jupyter_display.update({"text/plain": _render_to_text()}, raw=True)

def _rich_live_start(self, refresh: bool = False):
    from rich.live import _RefreshThread
    from IPython.display import display
    with self._lock:
        if self._started:
            return
        loopback = io.StringIO()
        # render to an offscreen console:
        loopback_console = rich.console.Console(
            force_jupyter=False,
            force_terminal=False,
            force_interactive=False,
            no_color=False,
            color_system="standard",
            file=loopback
        )
        self.loopback_console = loopback_console
        self.jupyter_display = display(display_id=True)

        self._started = True
        if refresh:
            try:
                self.refresh()
            except Exception:
                self.stop()
                raise
        if self.auto_refresh:
            self._refresh_thread = _RefreshThread(self, self.refresh_per_second)
            self._refresh_thread.start()

def _rich_live_stop(self):
    with self._lock:
        if not self._started:
            return
        self.console.clear_live()
        self._started = False
        if self.auto_refresh and self._refresh_thread is not None:
            self._refresh_thread.stop()
            self._refresh_thread = None
        self.refresh()

def setup_rich_notebook_hook():
    if rich.get_console().is_jupyter:
        # reconfigure not to use the jupyter console
        rich.reconfigure(
            color_system="standard",
            force_terminal=False,
            force_jupyter=False,
            force_interactive=True,
            no_color=False
        )
        rich.live.Live.refresh = _rich_live_refresh
        rich.live.Live.stop = _rich_live_stop
        rich.live.Live.start = _rich_live_start

def setup():
    jupyter = rich.get_console().is_jupyter
    # The driver path, for non-nix python environments
    if jupyter:
        setup_rich_notebook_hook()
    setup_logger(False)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")