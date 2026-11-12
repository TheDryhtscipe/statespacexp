"""Reference predictors for sanity checks.

Both predictors map int64 inputs [B, T] (any device) to float32 logits
[B, T, C] on the same device, so they plug straight into
:func:`sst.evaluation.evaluate`.

Oracle
------
:func:`oracle_predictor` recomputes the labels without the Cayley table or
the task's own label function:

* symmetric groups: compose the actual permutation arrays with numpy
  (``new_state[i] = state[g[i]]``, which is ``state . g``), then turn each
  permutation back into its lexicographic index with the Lehmer code;
* cyclic groups (parity, z3): cumulative sum mod n;
* flip-flop: for each position, find the index of the most recent write with
  a running maximum, then read that write's bit.

It returns one-hot logits and should score 1.0 everywhere.

Current-token-only
------------------
:func:`current_token_predictor` sees only the token at position t (not t
itself, not earlier tokens). The best prediction it can make:

* group tasks: predict s_t = g_t. At t = 1 this is exact. For t >= 2,
  s_{t-1} is uniform over G and independent of g_t, so s_t = s_{t-1} . g_t is
  uniform given g_t; every prediction is then right with probability 1/|G|.
  Predicting g_t is therefore optimal and its expected accuracy is 1 at t = 1
  and 1/|G| at t >= 2.
* flip-flop: on w0 or w1, predict the written bit. On r or i, predict 0.
  Only reads are scored, and a read says nothing about the stored bit, which
  is 0 or 1 with probability 1/2 each (writes are split evenly and position 1
  is a fair write). Expected accuracy at every scored position is 1/2.

:func:`current_token_expected_accuracy` returns these values.
"""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from sst.tasks import (
    FLIPFLOP_W0,
    FLIPFLOP_W1,
    FlipFlopTask,
    GroupTask,
    IntArray,
    Task,
)

Predictor = Callable[[torch.Tensor], torch.Tensor]


def _one_hot(classes: IntArray, num_classes: int, device: torch.device) -> torch.Tensor:
    idx = torch.from_numpy(classes).to(device)
    return torch.nn.functional.one_hot(idx, num_classes).to(torch.float32)


def lehmer_rank(perms: IntArray) -> IntArray:
    """Lexicographic rank of each row of ``perms`` [N, n] among permutations of range(n).

    Rank = sum_i c_i * (n - 1 - i)!, where c_i counts j > i with p[j] < p[i].
    This matches the order of ``itertools.permutations(range(n))``.
    """
    n = perms.shape[1]
    rank = np.zeros(perms.shape[0], dtype=np.int64)
    for i in range(n):
        smaller_after = (perms[:, i + 1 :] < perms[:, i : i + 1]).sum(axis=1)
        rank += smaller_after * math.factorial(n - 1 - i)
    return rank


def _perm_oracle(inputs: IntArray, perm_table: IntArray) -> IntArray:
    """Labels for a symmetric group by composing permutation arrays."""
    batch, length = inputs.shape
    n = perm_table.shape[1]
    state = np.tile(np.arange(n, dtype=np.int64), (batch, 1))
    out = np.empty_like(inputs)
    for t in range(length):
        g = perm_table[inputs[:, t]]  # [B, n], g[b, i] = g_t(i)
        state = np.take_along_axis(state, g, axis=1)  # state(g(i))
        out[:, t] = lehmer_rank(state)
    return out


def _flipflop_oracle(inputs: IntArray) -> IntArray:
    """Labels for flip-flop via the index of the most recent write."""
    is_write = (inputs == FLIPFLOP_W0) | (inputs == FLIPFLOP_W1)
    if not is_write[:, 0].all():
        raise ValueError("every flip-flop sequence must start with a write")
    positions = np.arange(inputs.shape[1], dtype=np.int64)[None, :]
    last_write: IntArray = np.maximum.accumulate(np.where(is_write, positions, -1), axis=1)
    written: IntArray = np.take_along_axis(inputs, last_write, axis=1)
    labels: IntArray = (written == FLIPFLOP_W1).astype(np.int64)
    return labels


def oracle_labels(task: Task, inputs: IntArray) -> IntArray:
    """Recompute labels for ``task`` by an independent route (see module docstring)."""
    if isinstance(task, GroupTask):
        perms = task.group.perms
        if perms is None:
            order = task.num_classes
            sums = np.cumsum(inputs, axis=1, dtype=np.int64)
            return np.mod(sums, order).astype(np.int64)
        return _perm_oracle(inputs, np.array(perms, dtype=np.int64))
    if isinstance(task, FlipFlopTask):
        return _flipflop_oracle(inputs)
    raise TypeError(f"no oracle for task type {type(task).__name__}")


def oracle_predictor(task: Task) -> Predictor:
    """One-hot logits of the exact labels, computed by :func:`oracle_labels`."""

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.detach().cpu().numpy().astype(np.int64)
        return _one_hot(oracle_labels(task, x), task.num_classes, inputs.device)

    return predict


def current_token_predictor(task: Task) -> Predictor:
    """Best prediction from the current token alone (see module docstring)."""
    if isinstance(task, GroupTask):

        def predict_group(inputs: torch.Tensor) -> torch.Tensor:
            x = inputs.detach().cpu().numpy().astype(np.int64)
            return _one_hot(x, task.num_classes, inputs.device)

        return predict_group
    if isinstance(task, FlipFlopTask):

        def predict_flipflop(inputs: torch.Tensor) -> torch.Tensor:
            x = inputs.detach().cpu().numpy().astype(np.int64)
            classes: npt.NDArray[np.int64] = (x == FLIPFLOP_W1).astype(np.int64)
            return _one_hot(classes, 2, inputs.device)

        return predict_flipflop
    raise TypeError(f"no current-token predictor for task type {type(task).__name__}")


def current_token_expected_accuracy(task: Task, position: int) -> float:
    """Expected accuracy of :func:`current_token_predictor` at a scored 1-based ``position``."""
    if position < 1:
        raise ValueError(f"position must be >= 1, got {position}")
    if isinstance(task, FlipFlopTask):
        # Only reads are scored, and a read carries no information.
        return 0.5
    if isinstance(task, GroupTask):
        return 1.0 if position == 1 else 1.0 / task.num_classes
    raise TypeError(f"no analytic value for task type {type(task).__name__}")
