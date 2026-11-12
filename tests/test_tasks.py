"""Label correctness and sampling for the state-tracking tasks."""

import numpy as np
import numpy.typing as npt
import pytest

from sst.groups import compose, symmetric_group
from sst.tasks import (
    FLIPFLOP_IGNORE,
    FLIPFLOP_READ,
    FLIPFLOP_W0,
    FLIPFLOP_W1,
    FlipFlopTask,
    GroupTask,
    Task,
    flipflop_labels,
    get_task,
    group_prefix_labels,
)
from sst.tasks import TASK_NAMES as REGISTERED_TASK_NAMES

TASK_NAMES = ["parity", "z3", "s3", "s5", "a5", "flipflop", "flipflop_sparse"]

# S3 elements in lexicographic order (index: permutation, written as (p(0), p(1), p(2))):
#   0: (0, 1, 2)  identity
#   1: (0, 2, 1)  swaps 1 and 2
#   2: (1, 0, 2)  swaps 0 and 1
#   3: (1, 2, 0)
#   4: (2, 0, 1)
#   5: (2, 1, 0)  swaps 0 and 2


def _arr(rows: list[list[int]]) -> npt.NDArray[np.int64]:
    return np.array(rows, dtype=np.int64)


# ---------------------------------------------------------------- group labels


def test_s3_hand_checked_example() -> None:
    # Tokens g1 = 2 = (1,0,2), g2 = 1 = (0,2,1), g3 = 2 = (1,0,2).
    # s1 = g1 = (1,0,2)                    -> index 2
    # s2 = g1 . g2, (g1 . g2)(i) = g1(g2(i)):
    #      i=0: g2(0)=0, g1(0)=1 -> 1
    #      i=1: g2(1)=2, g1(2)=2 -> 2
    #      i=2: g2(2)=1, g1(1)=0 -> 0       -> (1,2,0) = index 3
    # s3 = s2 . g3 = (1,2,0) . (1,0,2):
    #      i=0: g3(0)=1, s2(1)=2 -> 2
    #      i=1: g3(1)=0, s2(0)=1 -> 1
    #      i=2: g3(2)=2, s2(2)=0 -> 0       -> (2,1,0) = index 5
    table = symmetric_group(3).table
    labels = group_prefix_labels(_arr([[2, 1, 2]]), table)
    assert labels.tolist() == [[2, 3, 5]]


def test_s3_label_uses_a_dot_b_not_b_dot_a() -> None:
    g = symmetric_group(3)
    assert g.perms is not None
    index = {p: i for i, p in enumerate(g.perms)}
    found = False
    for a in range(6):
        for b in range(6):
            ab = index[compose(g.perms[a], g.perms[b])]
            ba = index[compose(g.perms[b], g.perms[a])]
            if ab != ba:
                found = True
                labels = group_prefix_labels(_arr([[a, b]]), g.table)
                assert labels[0, 1] == ab
                assert labels[0, 1] != ba
    assert found


@pytest.mark.parametrize("n", [3, 5])
def test_group_labels_match_brute_force_composition(n: int) -> None:
    g = symmetric_group(n)
    assert g.perms is not None
    index = {p: i for i, p in enumerate(g.perms)}
    rng = np.random.default_rng(123)
    inputs = rng.integers(0, len(g.perms), size=(16, 40), dtype=np.int64)
    labels = group_prefix_labels(inputs, g.table)
    for b in range(inputs.shape[0]):
        state = tuple(range(n))
        for t in range(inputs.shape[1]):
            perm = g.perms[int(inputs[b, t])]
            # Brute force: apply the new token first, then the running product.
            state = tuple(state[perm[i]] for i in range(n))
            assert labels[b, t] == index[state]


def test_parity_hand_written() -> None:
    task = get_task("parity")
    assert isinstance(task, GroupTask)
    labels = task.labels(_arr([[1, 0, 1, 1, 0], [0, 0, 0, 1, 1]]))
    assert labels.tolist() == [[1, 1, 0, 1, 1], [0, 0, 0, 1, 0]]


def test_z3_hand_written() -> None:
    task = get_task("z3")
    labels = task.labels(_arr([[2, 2, 1, 0, 1]]))
    assert labels.tolist() == [[2, 1, 2, 2, 0]]


@pytest.mark.parametrize("name,n", [("parity", 2), ("z3", 3)])
def test_cyclic_labels_match_running_sum(name: str, n: int) -> None:
    task = get_task(name)
    rng = np.random.default_rng(7)
    inputs = rng.integers(0, n, size=(8, 50), dtype=np.int64)
    labels = task.labels(inputs)
    for b in range(8):
        total = 0
        for t in range(50):
            total += int(inputs[b, t])
            assert labels[b, t] == total % n


# ----------------------------------------------------------- flip-flop labels


def test_flipflop_hand_written() -> None:
    w0, w1, r, i = FLIPFLOP_W0, FLIPFLOP_W1, FLIPFLOP_READ, FLIPFLOP_IGNORE
    inputs = _arr([[w1, r, i, w0, r, w1, i], [w0, w0, r, r, w1, w1, r]])
    labels = flipflop_labels(inputs)
    assert labels.tolist() == [[1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1]]


def test_flipflop_labels_match_brute_force() -> None:
    task = FlipFlopTask(p_write=0.3)
    batch = task.sample(16, 60, np.random.default_rng(5))
    for b in range(16):
        bit = -1
        for t in range(60):
            tok = int(batch.inputs[b, t])
            if tok == FLIPFLOP_W0:
                bit = 0
            elif tok == FLIPFLOP_W1:
                bit = 1
            assert bit in (0, 1)
            assert batch.targets[b, t] == bit


def test_flipflop_labels_reject_sequence_not_starting_with_write() -> None:
    with pytest.raises(ValueError):
        flipflop_labels(_arr([[FLIPFLOP_READ, FLIPFLOP_W0]]))


# ------------------------------------------------------------------- sampling


@pytest.mark.parametrize("name", TASK_NAMES)
def test_sample_shapes_dtypes_and_mask(name: str) -> None:
    task = get_task(name)
    batch = task.sample(5, 17, np.random.default_rng(0))
    assert batch.inputs.shape == (5, 17)
    assert batch.targets.shape == (5, 17)
    assert batch.mask.shape == (5, 17)
    assert batch.inputs.dtype == np.int64
    assert batch.targets.dtype == np.int64
    assert batch.mask.dtype == np.bool_
    assert batch.inputs.min() >= 0 and batch.inputs.max() < task.vocab_size
    assert batch.targets.min() >= 0 and batch.targets.max() < task.num_classes
    np.testing.assert_array_equal(batch.targets, task.labels(batch.inputs))


@pytest.mark.parametrize("name", ["parity", "z3", "s3", "s5"])
def test_group_tasks_score_every_position(name: str) -> None:
    batch = get_task(name).sample(5, 17, np.random.default_rng(0))
    assert batch.mask.all()


@pytest.mark.parametrize("name", ["flipflop", "flipflop_sparse"])
def test_flipflop_mask_is_reads_only(name: str) -> None:
    batch = get_task(name).sample(50, 80, np.random.default_rng(4))
    np.testing.assert_array_equal(batch.mask, batch.inputs == FLIPFLOP_READ)
    assert batch.mask.any()
    assert not batch.mask.all()


def test_flipflop_mask_hand_written() -> None:
    w0, w1, r, i = FLIPFLOP_W0, FLIPFLOP_W1, FLIPFLOP_READ, FLIPFLOP_IGNORE
    inputs = _arr([[w1, r, i, w0, r, w1, i]])
    mask = FlipFlopTask().mask(inputs)
    assert mask.tolist() == [[False, True, False, False, True, False, False]]
    # Labels stay defined at every position.
    assert FlipFlopTask().labels(inputs).tolist() == [[1, 1, 1, 0, 0, 1, 1]]


@pytest.mark.parametrize(
    "name,vocab,classes,chance",
    [
        ("parity", 2, 2, 1 / 2),
        ("z3", 3, 3, 1 / 3),
        ("s3", 6, 6, 1 / 6),
        ("s5", 120, 120, 1 / 120),
        ("flipflop", 4, 2, 1 / 2),
        ("flipflop_sparse", 4, 2, 1 / 2),
    ],
)
def test_task_metadata(name: str, vocab: int, classes: int, chance: float) -> None:
    task: Task = get_task(name)
    assert task.name == name
    assert task.vocab_size == vocab
    assert task.num_classes == classes
    assert task.chance_accuracy == pytest.approx(chance)


@pytest.mark.parametrize("name", ["parity", "z3", "s3", "s5"])
def test_group_inputs_cover_all_elements_uniformly(name: str) -> None:
    task = get_task(name)
    batch = task.sample(200, 300, np.random.default_rng(1))
    counts = np.bincount(batch.inputs.ravel(), minlength=task.vocab_size)
    expected = batch.inputs.size / task.vocab_size
    assert counts.min() > 0.8 * expected
    assert counts.max() < 1.2 * expected
    assert counts[0] > 0  # identity included


def test_flipflop_sampling_distribution() -> None:
    task = FlipFlopTask(p_write=0.1)
    batch = task.sample(400, 250, np.random.default_rng(2))
    first = batch.inputs[:, 0]
    assert np.isin(first, [FLIPFLOP_W0, FLIPFLOP_W1]).all()
    assert 0.4 < float(np.mean(first == FLIPFLOP_W1)) < 0.6
    rest = batch.inputs[:, 1:]
    frac = np.bincount(rest.ravel(), minlength=4) / rest.size
    assert frac[FLIPFLOP_W0] == pytest.approx(0.05, abs=0.005)
    assert frac[FLIPFLOP_W1] == pytest.approx(0.05, abs=0.005)
    assert frac[FLIPFLOP_READ] == pytest.approx(0.45, abs=0.01)
    assert frac[FLIPFLOP_IGNORE] == pytest.approx(0.45, abs=0.01)


def test_flipflop_p_write_parameter_is_used() -> None:
    batch = FlipFlopTask(p_write=0.5).sample(200, 200, np.random.default_rng(3))
    rest = batch.inputs[:, 1:]
    write_frac = float(np.isin(rest, [FLIPFLOP_W0, FLIPFLOP_W1]).mean())
    assert write_frac == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5])
def test_flipflop_rejects_bad_p_write(p: float) -> None:
    with pytest.raises(ValueError):
        FlipFlopTask(p_write=p)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_same_seed_same_batch(name: str) -> None:
    task = get_task(name)
    a = task.sample(4, 30, np.random.default_rng(11))
    b = task.sample(4, 30, np.random.default_rng(11))
    c = task.sample(4, 30, np.random.default_rng(12))
    np.testing.assert_array_equal(a.inputs, b.inputs)
    np.testing.assert_array_equal(a.targets, b.targets)
    assert not np.array_equal(a.inputs, c.inputs)


def test_get_task_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_task("s4")


@pytest.mark.parametrize("name", TASK_NAMES)
def test_get_task_names_round_trip(name: str) -> None:
    assert name in REGISTERED_TASK_NAMES
    assert get_task(name).name == name


def test_registry_lists_exactly_the_expected_tasks() -> None:
    assert sorted(REGISTERED_TASK_NAMES) == sorted(TASK_NAMES)


@pytest.mark.parametrize("name,p_write", [("flipflop", 0.1), ("flipflop_sparse", 0.02)])
def test_flipflop_variants_p_write(name: str, p_write: float) -> None:
    task = get_task(name)
    assert isinstance(task, FlipFlopTask)
    assert task.p_write == p_write


def test_flipflop_task_default_name() -> None:
    assert FlipFlopTask(p_write=0.3).name == "flipflop"
