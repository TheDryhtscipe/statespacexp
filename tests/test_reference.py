"""Reference predictors: the oracle is exact; current-token-only matches its analytic value."""

import itertools

import numpy as np
import pytest
import torch

from sst.evaluation import evaluate, make_rng
from sst.reference import (
    current_token_expected_accuracy,
    current_token_predictor,
    lehmer_rank,
    oracle_labels,
    oracle_predictor,
)
from sst.tasks import FLIPFLOP_IGNORE, FLIPFLOP_READ, FLIPFLOP_W0, FLIPFLOP_W1, FlipFlopTask, get_task

TASK_NAMES = ["parity", "z3", "s3", "s5", "flipflop", "flipflop_sparse"]
L = 8


@pytest.mark.parametrize("n", [3, 5])
def test_lehmer_rank_matches_itertools_order(n: int) -> None:
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    np.testing.assert_array_equal(lehmer_rank(perms), np.arange(len(perms)))


def test_oracle_s3_hand_example() -> None:
    # Same sequence as tests/test_tasks.py::test_s3_hand_checked_example.
    task = get_task("s3")
    logits = oracle_predictor(task)(torch.tensor([[2, 1, 2]]))
    assert logits.argmax(-1).tolist() == [[2, 3, 5]]


def test_oracle_flipflop_hand_example() -> None:
    w0, w1, r, i = FLIPFLOP_W0, FLIPFLOP_W1, FLIPFLOP_READ, FLIPFLOP_IGNORE
    x = np.array([[w1, r, i, w0, r, w1, i]], dtype=np.int64)
    assert oracle_labels(FlipFlopTask(), x).tolist() == [[1, 1, 1, 0, 0, 1, 1]]


@pytest.mark.parametrize("name", TASK_NAMES)
def test_oracle_agrees_with_task_labels(name: str) -> None:
    task = get_task(name)
    batch = task.sample(32, 100, make_rng(name, "train", 0))
    np.testing.assert_array_equal(oracle_labels(task, batch.inputs), batch.targets)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_oracle_scores_one_everywhere(name: str) -> None:
    task = get_task(name)
    result = evaluate(oracle_predictor(task), task, train_length=L, num_sequences=16, batch_size=5, seed=0)
    for b in result.buckets:
        assert b.token_accuracy == 1.0
        assert b.sequence_accuracy == 1.0
        assert b.last_position_accuracy == 1.0


@pytest.mark.parametrize(
    "name,tol",
    [("parity", 0.02), ("z3", 0.02), ("s3", 0.015), ("s5", 0.004)],
)
def test_current_token_matches_analytic_value(name: str, tol: float) -> None:
    task = get_task(name)
    result = evaluate(
        current_token_predictor(task), task, train_length=L, num_sequences=2000, batch_size=500, seed=0
    )
    first, *rest = result.buckets
    later = current_token_expected_accuracy(task, 2)
    # In-distribution bucket holds position 1 (always right) and L - 1 later positions.
    assert first.token_accuracy == pytest.approx((1.0 + (L - 1) * later) / L, abs=tol)
    for b in rest:
        assert b.token_accuracy == pytest.approx(later, abs=tol)
        assert b.last_position_accuracy == pytest.approx(later, abs=4 * tol)


@pytest.mark.parametrize("name,tol", [("flipflop", 0.02), ("flipflop_sparse", 0.03)])
def test_current_token_flipflop_is_chance_on_reads(name: str, tol: float) -> None:
    # Only reads are scored, and a read carries no information about the bit.
    task = get_task(name)
    result = evaluate(
        current_token_predictor(task), task, train_length=L, num_sequences=2000, batch_size=500, seed=0
    )
    expected = current_token_expected_accuracy(task, 2)
    assert expected == 0.5
    for b in result.buckets:
        assert b.token_accuracy == pytest.approx(expected, abs=tol)
        assert b.last_position_accuracy == pytest.approx(expected, abs=4 * tol)


def test_current_token_analytic_values() -> None:
    assert current_token_expected_accuracy(get_task("parity"), 1) == 1.0
    assert current_token_expected_accuracy(get_task("parity"), 5) == 0.5
    assert current_token_expected_accuracy(get_task("s5"), 2) == pytest.approx(1 / 120)
    for position in (1, 2, 9):
        assert current_token_expected_accuracy(FlipFlopTask(p_write=0.1), position) == 0.5
        assert current_token_expected_accuracy(get_task("flipflop_sparse"), position) == 0.5


@pytest.mark.parametrize("name", TASK_NAMES)
def test_current_token_is_not_an_oracle(name: str) -> None:
    task = get_task(name)
    result = evaluate(current_token_predictor(task), task, train_length=L, num_sequences=64, batch_size=64, seed=1)
    assert result.buckets[-1].token_accuracy is not None
    assert result.buckets[-1].token_accuracy < 0.9
