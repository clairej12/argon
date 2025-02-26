from argon.data import DataStream, StreamBuilder
from argon.random import PRNGSequence
from argon.typing import Array, ArrayLike, PRNGKey
from argon.struct import struct

import argon.nn as nn
import argon.numpy as npx
import argon.transforms as agt
import argon.graph
import argon.tree
import argon.random

from flax.nnx import Optimizer

from typing import (
    Any, TypeVar, Callable, Generic
)
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

from rich.text import Text as RichText
from rich.progress import (
    Progress, ProgressColumn,
    TextColumn, BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn
)
from rich.style import Style

import itertools
import functools
import jax

import logging
logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
Batch = TypeVar("Batch")
OptState = Any
Vars = Any
Metrics = Any

class Loop(Generic[Sample]):
    def __init__(self,
            rng_key: jax.Array,
            data: DataStream[Sample],
            max_iterations: int,
            trace_dir: str | None,
            progress: Progress,
            show_epochs: bool):
        self.rng_key = rng_key
        self.data = data
        try: epoch_iterations = len(data)
        except TypeError: epoch_iterations = None
        self.epoch_iterations = epoch_iterations
        self.max_epochs = (max_iterations // epoch_iterations 
                           if epoch_iterations is not None else None)
        self.max_iterations = max_iterations
        self.progress = progress
        self.trace_dir = trace_dir
        self.show_epochs = (self.epoch_iterations is not None and self.max_iterations > self.epoch_iterations) and show_epochs

        if self.progress is not None:
            self.iteration_task = progress.add_task("Iteration", total=max_iterations)
            if self.show_epochs:
                self.epoch_task = progress.add_task("Epoch", total=self.max_epochs)
                self.epoch_iteration_task = progress.add_task("Epoch Iteration", total=epoch_iterations)
        else:
            self.iteration_task = None
            self.epoch_task = None
            self.epoch_iteration_task = None

    def epochs(self) -> "Iterator[Epoch[Sample]]":
        if self.progress:
            self.progress.reset(self.iteration_task, total=self.max_iterations)
            if self.show_epochs:
                self.progress.reset(self.epoch_task, total=self.max_epochs)
                self.progress.reset(self.epoch_iteration_task, total=self.epoch_iterations)

        iterations = 0
        rng = PRNGSequence(self.rng_key) if self.rng_key is not None else None
        try:
            for e in itertools.count():
                epoch_iterations = self.max_iterations - iterations
                if self.epoch_iterations is not None:
                    epoch_iterations = min(epoch_iterations, self.epoch_iterations)
                sk = next(rng) if rng is not None else None
                yield Epoch(self, sk, e, iterations, epoch_iterations)
                iterations = iterations + epoch_iterations
                if self.progress and self.show_epochs:
                    self.progress.advance(self.epoch_task)
                if iterations >= self.max_iterations:
                    break
        finally:
            if self.progress:
                self.progress.refresh()

class Epoch(Generic[Sample]):
    def __init__(self, loop: Loop[Sample], rng_key: jax.Array,
                    epoch, prev_iterations, epoch_iterations):
        self.rng_key = rng_key
        self.loop = loop
        self.num = epoch
        self.prev_iterations = prev_iterations
        self.epoch_iterations = epoch_iterations

    @property
    def data(self):
        return self.loop.data
    
    def steps(self) -> "Iterator[Step[Sample]]":
        prev_iterations = self.prev_iterations
        if self.loop.progress and self.loop.show_epochs:
            self.loop.progress.reset(
                self.loop.epoch_iteration_task, total=self.epoch_iterations
            )
        rng = PRNGSequence(self.rng_key) if self.rng_key is not None else None
        for i in range(self.epoch_iterations):
            total_iter = prev_iterations + i
            with jax.profiler.StepTraceAnnotation("step", step_num=total_iter):
                with jax.profiler.TraceAnnotation("data_fetch"):
                    data = self.loop.data
                    if not data.has_next():
                        data.reset()
                    if not data.has_next(): raise ValueError("Unable to reset stream!")
                    t = time.time()
                    batch = data.next()
                    self.loop.data = data
                    sk = next(rng) if rng is not None else None
                with jax.profiler.TraceAnnotation("run_step"):
                    yield Step(batch, sk, self.num, 
                        i, total_iter)

            if self.loop.progress:
                if self.loop.show_epochs:
                    self.loop.progress.advance(self.loop.epoch_iteration_task)
                self.loop.progress.advance(self.loop.iteration_task)

class Step(Generic[Sample]):
    def __init__(self, batch : Sample, rng_key: jax.Array, epoch, epoch_iteration, iteration):
        self.rng_key = rng_key
        self.batch = batch
        self.epoch = epoch
        self.epoch_iteration = epoch_iteration
        self.iteration = iteration
    
    @property
    def num(self):
        return self.iteration

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return RichText(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )

@contextmanager
def loop(data : StreamBuilder[Sample], *, 
         iterations, rng_key=None, 
         progress=True, show_epochs=True,
         log_compiles=False, trace=False) -> Iterator[Loop[Sample]]:
    with data.build() as stream:
        if progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(finished_style=Style(color="green")),
                MofNColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                refresh_per_second=1
            )
            progress_ctx = progress
        else: 
            progress = None
            progress_ctx = nullcontext()
        if log_compiles: compile_logger = jax.log_compiles()
        else: compile_logger = nullcontext()
        if trace:
            trace_dir = Path("/tmp/jax-traces")#  / time.strftime("%Y_%m_%d-%H_%M_%S")
            trace_dir.mkdir(exist_ok=True, parents=True)
        else:
            trace_dir = None
        loop = Loop(
            rng_key,
            stream,
            iterations,
            progress=progress,
            show_epochs=show_epochs,
            trace_dir=trace_dir
        )
        with progress_ctx, compile_logger:
            yield loop

@struct(frozen=True)
class LossOutput:
    loss: ArrayLike | None = None
    metrics: Metrics = None

LossFn = Callable[[nn.Module, PRNGKey, Sample], LossOutput]

@agt.jit(static_argnums=(0,))
def _batched_loss(loss_fn: LossFn, model: nn.Module, rng_key: PRNGKey, batch: Sample):
    N = argon.tree.axis_size(batch)
    rng_keys = None
    if rng_key is not None:
        rng_keys = argon.random.split(rng_key, N)
    outputs = agt.vmap(loss_fn, in_axes=(None, 0, 0))(model, rng_keys, batch)
    loss = npx.mean(outputs.loss)
    metrics = argon.tree.map(npx.mean, outputs.metrics)
    return LossOutput(loss=loss, metrics=metrics)

def batch_loss(loss_fn: LossFn):
    return functools.partial(_batched_loss, loss_fn)

class TrainStep(Step[Sample], Generic[Sample]):
    def __init__(self, 
                    step: Step[Sample], metrics: Metrics, sys_metrics: Metrics,
                    graph_def: argon.graph.GraphDef, graph_state: argon.graph.GraphLeaves,
                    model: nn.Module, optimizer: Optimizer,
                    gradient: argon.graph.GraphLeaves | None = None,
                    grad_variance: Array | None = None
                 ):
        super().__init__(step.batch, step.rng_key, step.epoch, step.epoch_iteration, step.iteration)
        self.metrics = metrics
        self.sys_metrics = sys_metrics
        self.gradient = gradient
        self.gradient_variance = grad_variance

        self._graph_def = graph_def
        self._graph_state = graph_state
        self._model = model
        self._optimizer = optimizer
    
    def realize(self):
        argon.graph.update((self._model, self._optimizer), self._graph_state)
        # set the model to eval mode
        self._model.eval()
    
    def update(self, model: nn.Module | None, optimizer: Optimizer | None):
        model = self._model if model is None else model
        optimizer = self._optimizer if optimizer is None else optimizer
        model.train()
        graphdef, state = argon.graph.split((model, optimizer))
        self._graph_def = graphdef
        self._graph_state = state

def train_model(data: StreamBuilder[Sample],
                model: nn.Module, optimizer: Optimizer,
                loss: LossFn,
                *,
                iterations: int, rng_key: PRNGKey | None = None,
                store_gradient: bool = False,
                store_gradient_variance: bool = False,
                is_batch_loss: bool = False):
    # Set the model to training mode
    model.train()
    _loss = loss

    graphdef, state = argon.graph.split((model, optimizer))

    # For speed, we use the jax jit
    @functools.partial(jax.jit, donate_argnums=(1,))
    def train_step(graphdef: argon.graph.GraphDef, 
             state: argon.graph.GraphLeaves, 
             rng_key: PRNGKey | None, batch: Sample):
        model, optimizer = argon.graph.merge(graphdef, state)
        if store_gradient_variance:
            assert not is_batch_loss, "Cannot compute gradient variance with batched loss"
            def loss_fn(model, rng_key, batch):
                output = _loss(model, rng_key, batch)
                return output.loss, output.metrics
            batched_loss = agt.vmap(agt.grad(loss_fn, has_aux=True), in_axes=(None, 0, 0))
            N = argon.tree.axis_size(batch)
            rngs = argon.random.split(rng_key, N)
            grads_raw, metrics = batched_loss(model, rngs, batch)
            grads = argon.tree.map(lambda x: npx.mean(x,axis=0), grads_raw)
            grad_variance = argon.tree.reduce(npx.add, argon.tree.map(
                lambda x, m: npx.sum(npx.square(x - m[None,...])) / N, 
                grads_raw, grads
            ))
        else:
            @agt.jit
            def loss_fn(model, rng_key, batch):
                if not is_batch_loss: loss = batch_loss(_loss)
                else: loss = _loss
                output = loss(model, rng_key, batch)
                return output.loss, output.metrics
            grads, metrics = agt.grad(loss_fn, has_aux=True)(model, rng_key, batch)
            grad_variance = None
        optimizer.update(grads)
        graphdef, state = argon.graph.split((model, optimizer))
        return graphdef, state, metrics, (grads if store_gradient else None), grad_variance

    t = time.time()
    def _time():
        nonlocal t
        nt = time.time()
        dt = nt - t
        t = nt
        return dt
    eval_time, data_load_time, opt_time = 0.,0.,0.
    with loop(data, iterations=iterations, rng_key=rng_key) as l:
        for epoch in l.epochs():
            for step in epoch.steps():
                sys_metrics = {
                    "data_load_time": data_load_time,
                    "opt_time": opt_time,
                    "eval_time": eval_time,
                    "step_time": data_load_time + opt_time + eval_time
                }
                data_load_time = _time()
                graphdef, state, metrics, grads, grad_variance = train_step(
                    graphdef, state, step.rng_key, step.batch
                )
                opt_time = _time()
                tstep = TrainStep(step, metrics, sys_metrics,
                        graphdef, state, model, optimizer, grads, grad_variance)
                yield tstep
                graphdef, state = tstep._graph_def, tstep._graph_state
                eval_time = _time()
    # Restore the model, optimizer state
    argon.graph.update((model, optimizer), state)
    model.eval()

import time