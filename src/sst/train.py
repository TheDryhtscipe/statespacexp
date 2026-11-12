"""Training runner: one :class:`~sst.config.RunConfig` in, one :class:`~sst.results.RunResult` out.

Protocol
--------
* AdamW, betas (0.9, 0.999), weight decay ``config.weight_decay`` on every
  parameter with two or more dimensions (weight matrices, embeddings, learned
  initial states of shape [1, ...]); none on vectors and scalars (norm gains,
  biases, per-channel decay rates). Gradient-norm clipping at ``grad_clip``.
* Learning rate: linear warm-up over ``warmup_steps`` to ``lr``, then cosine
  decay to 0 at ``steps`` (:func:`lr_at`).
* Loss: :func:`masked_cross_entropy`, the mean over positions where
  ``TaskBatch.mask`` is True. Flip-flop scores reads only, so it trains on reads only.
* Data: ``make_rng(task, "train", seed)``. Batches are drawn ``data_chunk_steps``
  at a time: one ``task.sample(data_chunk_steps * batch_size, L, rng)`` call,
  split into consecutive batches. A longer run sees the same batches first.
* Model initialisation: ``torch.manual_seed(seed)`` before :func:`build_model`.
* Learning curves: mean train loss over each window of ``log_every`` steps;
  in-distribution val token accuracy (length L, ``curve_val_sequences``
  sequences of the val split) every ``val_every`` steps and at the end.
* Final evaluation: the val split (model selection) and the eval split
  (reporting) at ``eval_multiplier * L``, both with the fixed ``eval_seed``, so
  every run of a task sees the same val and eval sequences whatever its seed.
* Divergence: a non-finite train loss (checked at every log step, before every
  curve evaluation and at the end; training stops there) or non-finite logits in any evaluation gives status
  ``"diverged"`` with the stage, instead of an exception.

CUDA graphs
-----------
With ``config.cuda_graphs`` on a CUDA device, the whole step (forward, loss,
backward, clipping, AdamW with ``capturable=True`` and a tensor learning rate)
is captured once into a CUDA graph and replayed, which removes the per-kernel
launch cost that dominates the recurrent mixers' Python loops over time. The
first :data:`GRAPH_WARMUP_STEPS` steps run eagerly on a side stream (required
before capture) with the real batches, so the trajectory is the eager one.
:func:`graph_equivalence` checks a mixer's graphed losses against eager.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sst.config import RunConfig
from sst.evaluation import EvalResult, NonFiniteLogitsError, evaluate, make_rng
from sst.models import build_model, count_params
from sst.results import Curve, DivergedStage, RunResult, append_run
from sst.tasks import Task, get_task

__all__ = [
    "GRAPH_WARMUP_STEPS",
    "RunConfig",
    "Stepper",
    "TrainStream",
    "checkpoint_path",
    "graph_equivalence",
    "lr_at",
    "masked_cross_entropy",
    "run_and_record",
    "train",
]

GRAPH_WARMUP_STEPS = 3


def masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Mean cross-entropy over positions where ``mask`` is True; 0 if none are.

    ``logits`` [B, T, C], ``targets`` [B, T] int64, ``mask`` [B, T] bool. Unscored
    positions contribute nothing to the loss or the gradient, even if their logits
    are not finite. No data-dependent shapes, so the step stays graph-capturable.
    """
    c = logits.shape[-1]
    flat_mask = mask.reshape(-1)
    flat_logits = torch.where(flat_mask.unsqueeze(-1), logits.reshape(-1, c), 0.0)
    per = F.cross_entropy(flat_logits, targets.reshape(-1), reduction="none")
    per = torch.where(flat_mask, per, 0.0)
    return per.sum() / flat_mask.sum().clamp(min=1).to(per.dtype)


def lr_at(step: int, *, peak: float, warmup: int, total: int) -> float:
    """Learning rate for 0-based ``step``: linear warm-up then cosine decay to 0 at ``total``."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------ data


class TrainStream:
    """Training batches on ``device``, drawn in chunks; see the module docstring."""

    def __init__(self, task: Task, config: RunConfig, device: torch.device) -> None:
        self.task = task
        self.batch = config.batch_size
        self.length = config.train_length
        self.chunk = config.data_chunk_steps
        self.rng = make_rng(task.name, "train", config.seed)
        self.device = device

    def _upload(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.int64 | np.bool_]]) -> Tensor:
        t = torch.from_numpy(arr).reshape(self.chunk, self.batch, self.length)
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        while True:
            data = self.task.sample(self.chunk * self.batch, self.length, self.rng)
            inputs = self._upload(data.inputs)
            targets = self._upload(data.targets)
            mask = self._upload(data.mask)
            for i in range(self.chunk):
                yield inputs[i], targets[i], mask[i]


# ------------------------------------------------------------------ the step


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, list[nn.Parameter] | float]]:
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class Stepper:
    """Runs optimisation steps, eagerly or through a captured CUDA graph.

    Keeps on-device accumulators so the loop never synchronises with the host
    except when the caller reads them: ``loss_sum`` (sum of step losses since the
    last :meth:`read_window`), ``nonfinite`` (any non-finite loss so far) and
    ``last_loss``.
    """

    def __init__(self, model: nn.Module, config: RunConfig, device: torch.device, use_graphs: bool) -> None:
        if use_graphs and device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        self.model = model
        self.config = config
        self.device = device
        self.use_graphs = use_graphs
        self.params = [p for p in model.parameters() if p.requires_grad]
        groups = _param_groups(model, config.weight_decay)
        self.lr_tensor: Tensor | None = None
        if use_graphs:
            self.lr_tensor = torch.tensor(config.lr, device=device)
            self.opt = torch.optim.AdamW(groups, lr=self.lr_tensor, capturable=True)
        else:
            self.opt = torch.optim.AdamW(groups, lr=config.lr)
        self.loss_sum = torch.zeros((), device=device)
        self.last_loss = torch.zeros((), device=device)
        self.nonfinite = torch.zeros((), dtype=torch.bool, device=device)
        self.window_steps = 0
        self.calls = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static: tuple[Tensor, Tensor, Tensor] | None = None

    def _body(self, x: Tensor, y: Tensor, m: Tensor) -> None:
        logits: Tensor = self.model(x)
        loss = masked_cross_entropy(logits, y, m)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.params, self.config.grad_clip, foreach=True)
        self.opt.step()
        detached = loss.detach()
        self.last_loss.copy_(detached)
        self.loss_sum.add_(detached)
        self.nonfinite.logical_or_(~torch.isfinite(detached))

    def _set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            current = group["lr"]
            if isinstance(current, Tensor):
                current.fill_(lr)
            else:
                group["lr"] = lr

    def step(self, x: Tensor, y: Tensor, m: Tensor, lr: float) -> None:
        self._set_lr(lr)
        self.window_steps += 1
        self.calls += 1
        if not self.use_graphs:
            self.opt.zero_grad(set_to_none=True)
            self._body(x, y, m)
            return
        if self.calls <= GRAPH_WARMUP_STEPS:
            side = torch.cuda.Stream(self.device)  # type: ignore[no-untyped-call]
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                self.opt.zero_grad(set_to_none=True)
                self._body(x, y, m)
            torch.cuda.current_stream(self.device).wait_stream(side)
            return
        if self.graph is None:
            self.static = (x.clone(), y.clone(), m.clone())
            self.opt.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._body(*self.static)
            self.graph = graph
        assert self.static is not None
        sx, sy, sm = self.static
        sx.copy_(x, non_blocking=True)
        sy.copy_(y, non_blocking=True)
        sm.copy_(m, non_blocking=True)
        self.graph.replay()

    def read_window(self) -> tuple[float, bool]:
        """Mean loss over the steps since the last call, and whether any loss so far was non-finite.

        Synchronises with the device.
        """
        n = max(1, self.window_steps)
        mean = float((self.loss_sum / n).item())
        bad = bool(self.nonfinite.item())
        self.loss_sum.zero_()
        self.window_steps = 0
        return mean, bad


# ------------------------------------------------------------------ helpers


def _git_info() -> tuple[str, bool]:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def checkpoint_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / "checkpoints" / f"{run_id}.pt"


def _evaluate_model(model: nn.Module, task: Task, config: RunConfig, device: torch.device, *,
                    split: str, multiplier: int, num_sequences: int) -> EvalResult:
    model.eval()
    try:
        return evaluate(
            lambda x: model(x),
            task,
            train_length=config.train_length,
            num_sequences=num_sequences,
            batch_size=config.eval_batch_size,
            device=device,
            seed=config.eval_seed,
            max_multiplier=multiplier,
            split="val" if split == "val" else "eval",
        )
    finally:
        model.train()


@dataclass
class _Divergence:
    stage: DivergedStage
    step: int


def train(config: RunConfig, device: torch.device | str = "cpu", results_dir: Path | None = None) -> RunResult:
    """Train and evaluate one run; save its final ``state_dict`` under ``results_dir/checkpoints``.

    Does not write ``runs.jsonl``; see :func:`run_and_record`. ``results_dir=None`` skips the checkpoint.
    """
    dev = torch.device(device)
    if config.cuda_graphs and dev.type != "cuda":
        # cuda_graphs is part of run_id; an eager run must not be recorded under the graphed id.
        raise ValueError(f"cuda_graphs=True needs a cuda device, got {dev.type!r}")
    start = time.monotonic()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    task = get_task(config.task)
    torch.manual_seed(config.seed)
    model = build_model(config.model, task.vocab_size, task.num_classes).to(dev)
    model.train()
    use_graphs = config.cuda_graphs
    stepper = Stepper(model, config, dev, use_graphs)
    stream = iter(TrainStream(task, config, dev))

    train_curve: list[tuple[int, float | None]] = []
    val_curve: list[tuple[int, float | None]] = []
    divergence: _Divergence | None = None
    steps_done = 0
    for step in range(config.steps):
        x, y, m = next(stream)
        stepper.step(x, y, m, lr_at(step, peak=config.lr, warmup=config.warmup_steps, total=config.steps))
        steps_done = step + 1
        last = steps_done == config.steps
        if steps_done % config.log_every == 0 or last:
            mean, bad = stepper.read_window()
            train_curve.append((steps_done, mean if math.isfinite(mean) else None))
            if bad:
                divergence = _Divergence("train", steps_done)
                break
        if steps_done % config.val_every == 0 or last:
            if bool(stepper.nonfinite.item()):
                train_curve.append((steps_done, None))
                divergence = _Divergence("train", steps_done)
                break
            try:
                r = _evaluate_model(model, task, config, dev, split="val", multiplier=1,
                                    num_sequences=config.curve_val_sequences)
            except NonFiniteLogitsError:
                divergence = _Divergence("val_curve", steps_done)
                break
            val_curve.append((steps_done, r.buckets[0].token_accuracy))

    val: EvalResult | None = None
    ev: EvalResult | None = None
    if divergence is None:
        try:
            val = _evaluate_model(model, task, config, dev, split="val", multiplier=config.eval_multiplier,
                                  num_sequences=config.n_val_sequences)
        except NonFiniteLogitsError:
            divergence = _Divergence("final_val", steps_done)
    if divergence is None:
        try:
            ev = _evaluate_model(model, task, config, dev, split="eval", multiplier=config.eval_multiplier,
                                 num_sequences=config.n_eval_sequences)
        except NonFiniteLogitsError:
            divergence = _Divergence("final_eval", steps_done)
            val = None

    if divergence is None and results_dir is not None:
        path = checkpoint_path(results_dir, config.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)

    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    commit, dirty = _git_info()
    curve_t: Curve = tuple(train_curve)
    return RunResult(
        config=config,
        status="ok" if divergence is None else "diverged",
        diverged_stage=None if divergence is None else divergence.stage,
        diverged_at_step=None if divergence is None else divergence.step,
        steps_completed=steps_done,
        train_curve=curve_t,
        val_curve=tuple(val_curve),
        val=val,
        eval=ev,
        param_count=count_params(model),
        wall_time_s=time.monotonic() - start,
        peak_gpu_mem_bytes=peak,
        device=str(dev),
        cuda_graphs_used=use_graphs,
        git_commit=commit,
        git_dirty=dirty,
        torch_version=torch.__version__,
        finished_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def run_and_record(config: RunConfig, device: torch.device | str, results_dir: Path) -> RunResult:
    """:func:`train`, then append the record to ``results_dir/runs.jsonl``."""
    result = train(config, device, results_dir)
    append_run(results_dir / "runs.jsonl", result)
    return result


# ------------------------------------------------------------------ graph check


@dataclass(frozen=True)
class EquivalenceReport:
    eager_losses: tuple[float, ...]
    graph_losses: tuple[float, ...]
    eager_repeat_losses: tuple[float, ...]
    max_rel_diff: float
    noise_rel_diff: float
    passed: bool


def _loss_trajectory(config: RunConfig, device: torch.device, use_graphs: bool, steps: int) -> list[float]:
    task = get_task(config.task)
    torch.manual_seed(config.seed)
    model = build_model(config.model, task.vocab_size, task.num_classes).to(device)
    stepper = Stepper(model, config, device, use_graphs)
    stream = iter(TrainStream(task, config, device))
    losses: list[Tensor] = []
    for step in range(steps):
        x, y, m = next(stream)
        stepper.step(x, y, m, lr_at(step, peak=config.lr, warmup=config.warmup_steps, total=config.steps))
        losses.append(stepper.last_loss.clone())
    return [float(v) for v in torch.stack(losses).cpu().tolist()]


def _max_rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / max(abs(x), 1e-8) for x, y in zip(a, b, strict=True))


def graph_equivalence(config: RunConfig, device: torch.device | str = "cuda", steps: int = 20,
                      rtol: float = 1e-3) -> EquivalenceReport:
    """Compare the graphed and eager loss trajectories over ``steps`` steps from the same init and data.

    Also runs eager twice to measure the run-to-run noise floor of the GPU kernels
    (atomics make some backward kernels non-deterministic). Passes if every step's
    relative loss difference is within ``max(rtol, 10 * noise)``.
    """
    dev = torch.device(device)
    eager = _loss_trajectory(config, dev, False, steps)
    eager2 = _loss_trajectory(config, dev, False, steps)
    graph = _loss_trajectory(config, dev, True, steps)
    diff = _max_rel(eager, graph)
    noise = _max_rel(eager, eager2)
    passed = all(math.isfinite(v) for v in graph) and diff <= max(rtol, 10 * noise)
    return EquivalenceReport(tuple(eager), tuple(graph), tuple(eager2), diff, noise, passed)
