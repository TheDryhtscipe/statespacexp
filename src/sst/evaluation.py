"""Length-bucketed evaluation and the shared seeded-rng factory.

Random streams
--------------
:func:`make_rng` is the only place that turns ``(task_name, split, seed)``
into a generator. It builds ``np.random.SeedSequence`` from the entropy list
``[seed, SPLIT_CODES[split], task_code]``, where ``task_code`` is the first
8 bytes (little-endian) of SHA-256 of the task name. Different splits, seeds
or task names give different entropy, and SeedSequence maps different entropy
to independent streams. So training data (``split="train"``) never shares a
stream with evaluation data (``split="eval"``). ``"val"`` is for model
selection, so that ``"eval"`` is used only for reporting.

Buckets
-------
Training uses length ``L``. Evaluation uses length ``max_multiplier * L``
(default 8L). Positions are 1-based. With the default, the buckets are
``(0, L]``, ``(L, 2L]``, ``(2L, 4L]`` and ``(4L, 8L]``; the first is
in-distribution. For each bucket the result gives:

* ``token_accuracy``: correct masked positions / masked positions.
* ``sequence_accuracy``: among sequences with at least one masked position
  in the bucket, the fraction whose masked positions in the bucket are all
  correct.
* ``last_position_accuracy``: accuracy at exactly position ``end`` (L, 2L,
  4L, 8L), over the sequences where that position is masked. This is the
  usual "final-state accuracy at length n".

A metric with an empty denominator is ``None`` (``null`` in JSON).

The predicted class is ``argmax`` over the last logit dimension.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch

from sst.tasks import BoolArray, Task

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]

Split = Literal["train", "val", "eval"]
SPLIT_CODES: dict[str, int] = {"train": 0, "val": 1, "eval": 2}


def _task_code(task_name: str) -> int:
    return int.from_bytes(hashlib.sha256(task_name.encode("utf-8")).digest()[:8], "little")


def make_rng(task_name: str, split: Split, seed: int) -> np.random.Generator:
    """Return the generator for ``(task_name, split, seed)``; see the module docstring."""
    if split not in SPLIT_CODES:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(SPLIT_CODES)}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    entropy = [seed, SPLIT_CODES[split], _task_code(task_name)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def bucket_edges(train_length: int, max_multiplier: int = 8) -> tuple[int, ...]:
    """Return ``(0, L, 2L, 4L, ..., max_multiplier * L)``.

    Bucket k holds the 1-based positions ``edges[k] + 1 .. edges[k + 1]``.
    """
    if train_length < 1:
        raise ValueError(f"train_length must be >= 1, got {train_length}")
    if max_multiplier < 1 or max_multiplier & (max_multiplier - 1) != 0:
        raise ValueError(f"max_multiplier must be a power of two, got {max_multiplier}")
    edges = [0, train_length]
    while edges[-1] < max_multiplier * train_length:
        edges.append(2 * edges[-1])
    return tuple(edges)


@dataclass(frozen=True)
class BucketCounts:
    """Raw counts for one bucket; summed over batches."""

    token_correct: int
    token_scored: int
    seq_correct: int
    seq_scored: int
    last_correct: int
    last_scored: int

    def __add__(self, other: "BucketCounts") -> "BucketCounts":
        return BucketCounts(
            self.token_correct + other.token_correct,
            self.token_scored + other.token_scored,
            self.seq_correct + other.seq_correct,
            self.seq_scored + other.seq_scored,
            self.last_correct + other.last_correct,
            self.last_scored + other.last_scored,
        )


def bucket_counts(correct: BoolArray, mask: BoolArray, edges: Sequence[int]) -> list[BucketCounts]:
    """Count correct and scored positions per bucket.

    ``correct`` and ``mask`` are bool [B, T] with ``T == edges[-1]``. Bucket k
    covers 1-based positions ``edges[k] + 1 .. edges[k + 1]``, which is the
    0-based slice ``[edges[k], edges[k + 1])``. Its last position is 0-based
    index ``edges[k + 1] - 1``.
    """
    if correct.shape != mask.shape or correct.ndim != 2:
        raise ValueError(f"correct {correct.shape} and mask {mask.shape} must be equal 2-D shapes")
    if correct.shape[1] != edges[-1]:
        raise ValueError(f"sequence length {correct.shape[1]} != last edge {edges[-1]}")
    out: list[BucketCounts] = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        c = correct[:, lo:hi]
        m = mask[:, lo:hi]
        scored_any = m.any(axis=1)
        all_correct = (c | ~m).all(axis=1)
        last_c = correct[:, hi - 1]
        last_m = mask[:, hi - 1]
        out.append(
            BucketCounts(
                token_correct=int((c & m).sum()),
                token_scored=int(m.sum()),
                seq_correct=int((scored_any & all_correct).sum()),
                seq_scored=int(scored_any.sum()),
                last_correct=int((last_c & last_m).sum()),
                last_scored=int(last_m.sum()),
            )
        )
    return out


def _ratio(num: int, den: int) -> float | None:
    return num / den if den > 0 else None


@dataclass(frozen=True)
class BucketResult:
    """Metrics for the 1-based positions ``start + 1 .. end``."""

    start: int
    end: int
    token_accuracy: float | None
    num_scored: int
    sequence_accuracy: float | None
    num_sequences_scored: int
    last_position: int
    last_position_accuracy: float | None
    last_position_count: int


@dataclass(frozen=True)
class EvalResult:
    task: str
    train_length: int
    eval_length: int
    num_sequences: int
    seed: int
    buckets: tuple[BucketResult, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        buckets: list[JSONValue] = []
        for b in self.buckets:
            row: dict[str, JSONValue] = {}
            for key, value in asdict(b).items():
                if not isinstance(value, (int, float)) and value is not None:
                    raise TypeError(f"unexpected field type for {key}: {type(value)}")
                row[key] = value
            buckets.append(row)
        return {
            "task": self.task,
            "train_length": self.train_length,
            "eval_length": self.eval_length,
            "num_sequences": self.num_sequences,
            "seed": self.seed,
            "buckets": buckets,
        }


Predictor = Callable[[torch.Tensor], torch.Tensor]


def evaluate(
    predict: Predictor,
    task: Task,
    *,
    train_length: int = 64,
    num_sequences: int = 1024,
    batch_size: int = 128,
    device: torch.device | str = "cpu",
    seed: int = 0,
    max_multiplier: int = 8,
) -> EvalResult:
    """Score ``predict`` on ``num_sequences`` eval sequences of length ``max_multiplier * L``.

    ``predict`` maps int64 inputs [B, T] on ``device`` to logits [B, T, C]
    with ``C == task.num_classes``. It runs under ``torch.no_grad()``; putting
    a model in eval mode is the caller's job.

    The whole eval set is drawn up front from ``make_rng(task.name, "eval",
    seed)`` with one ``task.sample`` call, so the data and the result do not
    depend on ``batch_size``.
    """
    if num_sequences < 1 or batch_size < 1:
        raise ValueError("num_sequences and batch_size must be >= 1")
    edges = bucket_edges(train_length, max_multiplier)
    eval_length = edges[-1]
    data = task.sample(num_sequences, eval_length, make_rng(task.name, "eval", seed))
    totals = [BucketCounts(0, 0, 0, 0, 0, 0) for _ in range(len(edges) - 1)]
    with torch.no_grad():
        for lo in range(0, num_sequences, batch_size):
            hi = min(lo + batch_size, num_sequences)
            inputs = torch.from_numpy(data.inputs[lo:hi]).to(device)
            logits = predict(inputs)
            expected_shape = (hi - lo, eval_length, task.num_classes)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(f"logits shape {tuple(logits.shape)} != expected {expected_shape}")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("logits contain NaN or inf")
            preds = logits.argmax(dim=-1).cpu().numpy()
            correct = preds == data.targets[lo:hi]
            batch_counts = bucket_counts(correct, data.mask[lo:hi], edges)
            totals = [t + c for t, c in zip(totals, batch_counts, strict=True)]
    buckets = tuple(
        BucketResult(
            start=edges[k],
            end=edges[k + 1],
            token_accuracy=_ratio(c.token_correct, c.token_scored),
            num_scored=c.token_scored,
            sequence_accuracy=_ratio(c.seq_correct, c.seq_scored),
            num_sequences_scored=c.seq_scored,
            last_position=edges[k + 1],
            last_position_accuracy=_ratio(c.last_correct, c.last_scored),
            last_position_count=c.last_scored,
        )
        for k, c in enumerate(totals)
    )
    return EvalResult(
        task=task.name,
        train_length=train_length,
        eval_length=eval_length,
        num_sequences=num_sequences,
        seed=seed,
        buckets=buckets,
    )
