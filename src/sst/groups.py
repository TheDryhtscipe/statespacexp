"""Finite groups as explicit Cayley tables.

Conventions
-----------
Elements are integer indices 0 .. |G|-1. Index 0 is always the identity.

``table[a, b]`` is the index of the product ``a . b``.

Symmetric groups S_n: elements are the permutations of ``range(n)`` in the
order ``itertools.permutations(range(n))`` yields them (lexicographic), so
index 0 is the identity. A permutation ``p`` is stored as the tuple
``(p(0), p(1), ..., p(n-1))``. Composition is function composition, applying
the right-hand factor first::

    (a . b)(i) = a(b(i))

The running state of a sequence of tokens g_1, ..., g_t is the left-to-right
product ``s_t = g_1 . g_2 . ... . g_t``, which satisfies
``s_t = s_{t-1} . g_t``, so ``s_t = table[s_{t-1}, g_t]``.

Alternating groups A_n: the even permutations of ``range(n)``, kept in the
same lexicographic order as in S_n (so index 0 is still the identity) and
composed by the same rule.

Cyclic groups Z_n: element k is the residue k, and ``a . b = (a + b) mod n``.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Perm = tuple[int, ...]


@dataclass(frozen=True, eq=False)
class Group:
    """A finite group given by its Cayley table.

    ``perms`` holds the permutation for each index for symmetric groups and is
    ``None`` for cyclic groups.
    """

    name: str
    table: npt.NDArray[np.int64]
    identity: int
    perms: tuple[Perm, ...] | None


def compose(a: Perm, b: Perm) -> Perm:
    """Return ``a . b``, defined by ``(a . b)(i) = a(b(i))`` (apply ``b`` first)."""
    if len(a) != len(b):
        raise ValueError("permutations must have the same length")
    return tuple(a[b[i]] for i in range(len(b)))


def cyclic_group(n: int) -> Group:
    """Z_n under addition mod n."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    idx = np.arange(n, dtype=np.int64)
    table = (idx[:, None] + idx[None, :]) % n
    table.setflags(write=False)
    return Group(name=f"Z{n}", table=table, identity=0, perms=None)


def _permutation_group(name: str, perms: tuple[Perm, ...]) -> Group:
    """Cayley table of ``perms`` (closed under ``compose``, identity first)."""
    index = {p: i for i, p in enumerate(perms)}
    size = len(perms)
    table = np.empty((size, size), dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            table[i, j] = index[compose(a, b)]
    table.setflags(write=False)
    return Group(name=name, table=table, identity=index[tuple(range(len(perms[0])))], perms=perms)


def symmetric_group(n: int) -> Group:
    """S_n with permutations in lexicographic order and ``(a . b)(i) = a(b(i))``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return _permutation_group(f"S{n}", tuple(itertools.permutations(range(n))))


def is_even(p: Perm) -> bool:
    """True if ``p`` has an even number of inversions."""
    inversions = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return inversions % 2 == 0


def alternating_group(n: int) -> Group:
    """A_n: the even permutations of S_n, in lexicographic order, same composition."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    perms = tuple(p for p in itertools.permutations(range(n)) if is_even(p))
    return _permutation_group(f"A{n}", perms)


def cayley(group: Group) -> npt.NDArray[np.int64]:
    """Return a writable copy of the Cayley table: ``out[a, b]`` is ``a . b``."""
    return group.table.copy()


def identity_index(group: Group) -> int:
    """Index of the identity element (always 0 for the groups built here)."""
    return group.identity
