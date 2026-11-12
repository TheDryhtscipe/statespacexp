"""Cayley tables must be valid groups, and composition must follow the documented convention."""

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from sst.groups import Group, cayley, compose, cyclic_group, identity_index, symmetric_group


def _all_triples(n: int) -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in range(n) for b in range(n) for c in range(n)]


def _check_group_axioms(table: npt.NDArray[np.int64], identity: int, triples: list[tuple[int, int, int]]) -> None:
    n = table.shape[0]
    assert table.shape == (n, n)
    assert table.dtype == np.int64
    # Closure, and each row/column is a permutation of the elements (Latin square).
    for i in range(n):
        assert sorted(table[i].tolist()) == list(range(n))
        assert sorted(table[:, i].tolist()) == list(range(n))
    # Identity on both sides.
    assert table[identity].tolist() == list(range(n))
    assert table[:, identity].tolist() == list(range(n))
    # Every element has a two-sided inverse.
    for a in range(n):
        inverses = [b for b in range(n) if table[a, b] == identity]
        assert len(inverses) == 1
        assert table[inverses[0], a] == identity
    # Associativity.
    for a, b, c in triples:
        assert table[table[a, b], c] == table[a, table[b, c]]


@pytest.mark.parametrize("n", [2, 3])
def test_cyclic_group_is_addition_mod_n(n: int) -> None:
    g = cyclic_group(n)
    table = cayley(g)
    assert identity_index(g) == 0
    for a in range(n):
        for b in range(n):
            assert table[a, b] == (a + b) % n
    _check_group_axioms(table, 0, _all_triples(n))


def test_s3_is_valid_group_all_triples() -> None:
    g = symmetric_group(3)
    table = cayley(g)
    assert table.shape == (6, 6)
    assert identity_index(g) == 0
    _check_group_axioms(table, 0, _all_triples(6))


def test_s5_is_valid_group_sampled_triples() -> None:
    g = symmetric_group(5)
    table = cayley(g)
    assert table.shape == (120, 120)
    assert identity_index(g) == 0
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 120, size=(5000, 3))
    triples = [(int(a), int(b), int(c)) for a, b, c in draws]
    _check_group_axioms(table, 0, triples)


@pytest.mark.parametrize("n", [3, 5])
def test_permutations_enumerated_lexicographically(n: int) -> None:
    g = symmetric_group(n)
    assert g.perms == tuple(itertools.permutations(range(n)))
    assert g.perms is not None and g.perms[0] == tuple(range(n))


def test_compose_applies_right_argument_first() -> None:
    # a = (0 1): swaps 0 and 1.  b = (1 2): swaps 1 and 2.
    a = (1, 0, 2)
    b = (0, 2, 1)
    # (a . b)(i) = a(b(i)):
    #   i=0: b(0)=0, a(0)=1 -> 1
    #   i=1: b(1)=2, a(2)=2 -> 2
    #   i=2: b(2)=1, a(1)=0 -> 0
    assert compose(a, b) == (1, 2, 0)
    # The reverse order gives a different permutation:
    #   (b . a)(i) = b(a(i)): i=0 -> b(1)=2; i=1 -> b(0)=0; i=2 -> b(2)=1
    assert compose(b, a) == (2, 0, 1)


@pytest.mark.parametrize("n", [3, 5])
def test_cayley_entries_match_compose(n: int) -> None:
    g = symmetric_group(n)
    assert g.perms is not None
    table = cayley(g)
    index = {p: i for i, p in enumerate(g.perms)}
    for i, a in enumerate(g.perms):
        for j, b in enumerate(g.perms):
            expected = tuple(a[b[k]] for k in range(n))
            assert table[i, j] == index[expected]


def test_s3_is_not_abelian() -> None:
    table = cayley(symmetric_group(3))
    assert not np.array_equal(table, table.T)


def test_cayley_returns_copy() -> None:
    g: Group = symmetric_group(3)
    t = cayley(g)
    t[0, 0] = 99
    assert cayley(g)[0, 0] == 0
