"""State-tracking tasks with dense per-position labels.

Every task produces a :class:`TaskBatch` of int64 ``inputs`` [B, T], int64
``targets`` [B, T] and a bool ``mask`` [B, T] (True = the position is scored).
Labels are dense: there is a target at every position.

Input sampling and label computation are kept apart. ``Task.sample_inputs``
draws tokens; ``Task.labels`` is a pure function of the tokens. The label
functions :func:`group_prefix_labels` and :func:`flipflop_labels` are
module-level so they can be tested on their own.

Tasks
-----
``parity``   Z2. Tokens are bits; target_t = x_1 XOR ... XOR x_t.
``z3``       Z3. Tokens are 0, 1, 2; target_t = (x_1 + ... + x_t) mod 3.
``s3``/``s5`` Symmetric groups. Token k is the k-th permutation in
             lexicographic order (see :mod:`sst.groups`); target_t is the
             index of s_t = g_1 . g_2 . ... . g_t, where (a . b)(i) = a(b(i)).
``a5``       Alternating group A5 (60 even permutations of 5), lexicographic
             order and the same composition as s3/s5.
``flipflop`` Tokens w0, w1, r, i (write 0, write 1, read, ignore). Target_t
             is the most recently written bit, including a write at t.
             Only reads are scored (mask True exactly at r tokens).
             ``flipflop`` has p_write 0.1; ``flipflop_sparse`` has 0.02.

For all group tasks the token id is the element's index in the Cayley table,
so ``vocab_size == num_classes == |G|``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sst.groups import Group, alternating_group, cyclic_group, symmetric_group

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

FLIPFLOP_W0 = 0
FLIPFLOP_W1 = 1
FLIPFLOP_READ = 2
FLIPFLOP_IGNORE = 3


@dataclass(frozen=True, eq=False)
class TaskBatch:
    inputs: IntArray
    targets: IntArray
    mask: BoolArray


def _check_inputs(inputs: IntArray) -> None:
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be 2-D [B, T], got shape {inputs.shape}")
    if inputs.dtype != np.int64:
        raise ValueError(f"inputs must be int64, got {inputs.dtype}")


def group_prefix_labels(inputs: IntArray, table: IntArray) -> IntArray:
    """Return the running product s_t = g_1 . g_2 . ... . g_t at every position.

    ``inputs[b, t]`` is the element index of g_{t+1} (0-based t). The loop uses
    ``s_t = table[s_{t-1}, g_t]`` with ``s_1 = g_1``, which is the left-to-right
    product under the convention of :mod:`sst.groups`.
    """
    _check_inputs(inputs)
    n = table.shape[0]
    if inputs.size and (inputs.min() < 0 or inputs.max() >= n):
        raise ValueError(f"group tokens must be in [0, {n})")
    out = np.empty_like(inputs)
    state = inputs[:, 0].copy()
    out[:, 0] = state
    for t in range(1, inputs.shape[1]):
        state = table[state, inputs[:, t]]
        out[:, t] = state
    return out


def flipflop_labels(inputs: IntArray) -> IntArray:
    """Return the most recently written bit at every position.

    A write at position t counts for position t. Every sequence must start
    with a write, so the stored bit is always defined.
    """
    _check_inputs(inputs)
    if inputs.size and (inputs.min() < 0 or inputs.max() > FLIPFLOP_IGNORE):
        raise ValueError("flip-flop tokens must be in [0, 3]")
    first = inputs[:, 0]
    if not np.all((first == FLIPFLOP_W0) | (first == FLIPFLOP_W1)):
        raise ValueError("every flip-flop sequence must start with a write")
    out = np.empty_like(inputs)
    bit = first - FLIPFLOP_W0
    out[:, 0] = bit
    for t in range(1, inputs.shape[1]):
        tok = inputs[:, t]
        bit = np.where(tok == FLIPFLOP_W0, 0, np.where(tok == FLIPFLOP_W1, 1, bit))
        out[:, t] = bit
    return out


class Task(ABC):
    """A state-tracking task: samples token sequences and labels them."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def chance_accuracy(self) -> float:
        """Accuracy of the best constant prediction at a scored position t >= 2."""

    @abstractmethod
    def sample_inputs(self, batch: int, length: int, rng: np.random.Generator) -> IntArray:
        """Draw int64 tokens of shape [batch, length]."""

    @abstractmethod
    def labels(self, inputs: IntArray) -> IntArray:
        """Compute int64 targets [B, T] from inputs; a pure function."""

    def mask(self, inputs: IntArray) -> BoolArray:
        """Which positions are scored. Every position, unless a task overrides it."""
        return np.ones(inputs.shape, dtype=np.bool_)

    def sample(self, batch: int, length: int, rng: np.random.Generator) -> TaskBatch:
        if batch < 1 or length < 1:
            raise ValueError(f"batch and length must be >= 1, got {batch}, {length}")
        inputs = self.sample_inputs(batch, length, rng)
        return TaskBatch(inputs=inputs, targets=self.labels(inputs), mask=self.mask(inputs))


class GroupTask(Task):
    """Running product over a finite group; tokens drawn uniformly from all elements."""

    def __init__(self, name: str, group: Group) -> None:
        self._name = name
        self.group = group

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return int(self.group.table.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.group.table.shape[0])

    @property
    def chance_accuracy(self) -> float:
        # With tokens uniform over G, s_t is uniform over G for every t >= 1.
        return 1.0 / self.num_classes

    def sample_inputs(self, batch: int, length: int, rng: np.random.Generator) -> IntArray:
        return rng.integers(0, self.vocab_size, size=(batch, length), dtype=np.int64)

    def labels(self, inputs: IntArray) -> IntArray:
        return group_prefix_labels(inputs, self.group.table)


class FlipFlopTask(Task):
    """Flip-flop language over {w0, w1, r, i}.

    Position 1 is w0 or w1 with equal probability. Every later token is a
    write with probability ``p_write`` (split evenly between w0 and w1),
    otherwise r or i with equal probability.

    Only reads are scored: the mask is True exactly at r tokens. The label
    (the stored bit) is still defined at every position. ``name`` lets the
    registry give each p_write setting its own name, and so its own rng
    stream in :func:`sst.evaluation.make_rng`.
    """

    def __init__(self, p_write: float = 0.1, name: str = "flipflop") -> None:
        if not 0.0 < p_write <= 1.0:
            raise ValueError(f"p_write must be in (0, 1], got {p_write}")
        self.p_write = p_write
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return 4

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def chance_accuracy(self) -> float:
        # The stored bit is 0 or 1 with probability 1/2 at every position.
        return 0.5

    def sample_inputs(self, batch: int, length: int, rng: np.random.Generator) -> IntArray:
        is_write = rng.random((batch, length)) < self.p_write
        is_write[:, 0] = True
        coin = rng.integers(0, 2, size=(batch, length), dtype=np.int64)
        # Writes: coin picks w0/w1. Others: coin picks r/i.
        tokens = np.where(is_write, FLIPFLOP_W0 + coin, FLIPFLOP_READ + coin)
        return tokens.astype(np.int64)

    def labels(self, inputs: IntArray) -> IntArray:
        return flipflop_labels(inputs)

    def mask(self, inputs: IntArray) -> BoolArray:
        """Score only read positions."""
        is_read: BoolArray = np.equal(inputs, FLIPFLOP_READ)
        return is_read


_REGISTRY: dict[str, Callable[[], Task]] = {
    "parity": lambda: GroupTask("parity", cyclic_group(2)),
    "z3": lambda: GroupTask("z3", cyclic_group(3)),
    "s3": lambda: GroupTask("s3", symmetric_group(3)),
    "s5": lambda: GroupTask("s5", symmetric_group(5)),
    "a5": lambda: GroupTask("a5", alternating_group(5)),
    "flipflop": lambda: FlipFlopTask(p_write=0.1, name="flipflop"),
    "flipflop_sparse": lambda: FlipFlopTask(p_write=0.02, name="flipflop_sparse"),
}

TASK_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def get_task(name: str) -> Task:
    """Build a registered task by name; ``get_task(name).name == name``."""
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"unknown task {name!r}; expected one of {TASK_NAMES}")
    return factory()
