"""A5: the even permutations of 5, in lexicographic order, with the S_n composition convention."""

import itertools

import numpy as np
import pytest

import numpy.typing as npt

from sst.groups import alternating_group, cayley, compose, identity_index
from sst.tasks import GroupTask, get_task, group_prefix_labels


def _check_group_axioms(table: npt.NDArray[np.int64], triples: list[tuple[int, int, int]]) -> None:
    n = table.shape[0]
    for i in range(n):
        assert sorted(table[i].tolist()) == list(range(n))
        assert sorted(table[:, i].tolist()) == list(range(n))
    assert table[0].tolist() == list(range(n))
    assert table[:, 0].tolist() == list(range(n))
    for a, b, c in triples:
        assert table[table[a, b], c] == table[a, table[b, c]]


def _is_even(p: tuple[int, ...]) -> bool:
    inversions = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return inversions % 2 == 0


def test_a5_elements_are_even_permutations_in_lexicographic_order() -> None:
    g = alternating_group(5)
    expected = tuple(p for p in itertools.permutations(range(5)) if _is_even(p))
    assert len(expected) == 60
    assert g.perms == expected
    assert g.perms[0] == (0, 1, 2, 3, 4)
    assert identity_index(g) == 0
    assert g.name == "A5"


def test_a5_is_valid_group_sampled_triples() -> None:
    g = alternating_group(5)
    table = cayley(g)
    assert table.shape == (60, 60)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 60, size=(5000, 3))
    _check_group_axioms(table, [(int(a), int(b), int(c)) for a, b, c in draws])


def test_a5_cayley_entries_match_compose() -> None:
    g = alternating_group(5)
    assert g.perms is not None
    index = {p: i for i, p in enumerate(g.perms)}
    for i, a in enumerate(g.perms):
        for j, b in enumerate(g.perms):
            assert g.table[i, j] == index[compose(a, b)]


def test_a5_is_not_abelian() -> None:
    table = cayley(alternating_group(5))
    assert not np.array_equal(table, table.T)


def test_a5_labels_match_brute_force_composition() -> None:
    g = alternating_group(5)
    assert g.perms is not None
    index = {p: i for i, p in enumerate(g.perms)}
    rng = np.random.default_rng(321)
    inputs = rng.integers(0, 60, size=(16, 40), dtype=np.int64)
    labels = group_prefix_labels(inputs, g.table)
    for b in range(inputs.shape[0]):
        state = tuple(range(5))
        for t in range(inputs.shape[1]):
            perm = g.perms[int(inputs[b, t])]
            state = tuple(state[perm[i]] for i in range(5))
            assert labels[b, t] == index[state]


def test_a5_registered_task() -> None:
    task = get_task("a5")
    assert isinstance(task, GroupTask)
    assert task.name == "a5"
    assert task.vocab_size == 60
    assert task.num_classes == 60
    assert task.chance_accuracy == pytest.approx(1 / 60)
    batch = task.sample(4, 30, np.random.default_rng(0))
    assert batch.mask.all()
    np.testing.assert_array_equal(batch.targets, group_prefix_labels(batch.inputs, task.group.table))


def test_alternating_group_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        alternating_group(0)
