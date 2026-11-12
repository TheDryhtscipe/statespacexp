"""Length-bucketed evaluation: bucket boundaries, mask handling, rng separation."""

import json
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from sst.evaluation import (
    BucketCounts,
    EvalResult,
    NonFiniteLogitsError,
    Split,
    bucket_counts,
    bucket_edges,
    evaluate,
    make_rng,
)
from sst.tasks import FLIPFLOP_READ, BoolArray, IntArray, Task, get_task

Predictor = Callable[[torch.Tensor], torch.Tensor]


def _controlled_predictor(task: Task, wrong: Callable[[int], bool]) -> Predictor:
    """Predict the true label, except at 1-based positions p where wrong(p) is True.

    At a wrong position it predicts (label + 1) mod C, which is never correct.
    """

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.cpu().numpy().astype(np.int64)
        labels = task.labels(x)
        cols = np.array([wrong(p) for p in range(1, x.shape[1] + 1)], dtype=np.bool_)
        preds = np.where(cols[None, :], (labels + 1) % task.num_classes, labels)
        logits = torch.zeros(x.shape[0], x.shape[1], task.num_classes)
        logits.scatter_(2, torch.from_numpy(preds).unsqueeze(-1), 1.0)
        return logits.to(inputs.device)

    return predict


def _ba(rows: list[list[int]]) -> BoolArray:
    return np.array(rows, dtype=np.bool_)


# --------------------------------------------------------------- bucket edges


def test_bucket_edges_default() -> None:
    assert bucket_edges(64) == (0, 64, 128, 256, 512)
    assert bucket_edges(4) == (0, 4, 8, 16, 32)


def test_bucket_edges_rejects_non_power_of_two_multiplier() -> None:
    with pytest.raises(ValueError):
        bucket_edges(8, max_multiplier=6)


# ------------------------------------------------------- bucket_counts (pure)


def test_bucket_counts_hand_computed() -> None:
    # L = 2, positions 1..8, buckets (0,2], (2,4], (4,8].
    edges = (0, 2, 4, 8)
    correct = _ba(
        [
            [1, 1, 0, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    mask = np.ones_like(correct)
    counts = bucket_counts(correct, mask, edges)
    assert len(counts) == 3
    b1, b2, b3 = counts
    # Bucket (0,2]: positions 1,2.
    assert (b1.token_correct, b1.token_scored) == (3, 4)
    assert (b1.seq_correct, b1.seq_scored) == (1, 2)
    assert (b1.last_correct, b1.last_scored) == (1, 2)  # position 2
    # Bucket (2,4]: positions 3,4.
    assert (b2.token_correct, b2.token_scored) == (3, 4)
    assert (b2.seq_correct, b2.seq_scored) == (1, 2)
    assert (b2.last_correct, b2.last_scored) == (2, 2)  # position 4
    # Bucket (4,8]: positions 5..8.
    assert (b3.token_correct, b3.token_scored) == (7, 8)
    assert (b3.seq_correct, b3.seq_scored) == (1, 2)
    assert (b3.last_correct, b3.last_scored) == (1, 2)  # position 8


def test_bucket_counts_respects_mask() -> None:
    edges = (0, 2, 4)
    correct = _ba([[0, 1, 0, 0], [1, 1, 1, 1]])
    mask = _ba([[0, 1, 0, 0], [1, 1, 1, 0]])
    b1, b2 = bucket_counts(correct, mask, edges)
    # Row 0: the wrong position 1 is unscored, so bucket 1 is all correct.
    assert (b1.token_correct, b1.token_scored) == (3, 3)
    assert (b1.seq_correct, b1.seq_scored) == (2, 2)
    assert (b1.last_correct, b1.last_scored) == (2, 2)
    # Row 0 has nothing scored in bucket 2: it is left out of the sequence count.
    assert (b2.token_correct, b2.token_scored) == (1, 1)
    assert (b2.seq_correct, b2.seq_scored) == (1, 1)
    # Position 4 is unscored everywhere.
    assert (b2.last_correct, b2.last_scored) == (0, 0)


def test_bucket_counts_add() -> None:
    a = BucketCounts(1, 2, 3, 4, 5, 6)
    b = BucketCounts(10, 20, 30, 40, 50, 60)
    assert a + b == BucketCounts(11, 22, 33, 44, 55, 66)


def test_bucket_counts_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        bucket_counts(_ba([[1, 1, 1]]), _ba([[1, 1, 1]]), (0, 2, 4))


# ------------------------------------------------- evaluate, crafted predictors

L = 8


def test_correct_only_in_distribution() -> None:
    task = get_task("s3")
    result = evaluate(
        _controlled_predictor(task, lambda p: p > L),
        task,
        train_length=L,
        num_sequences=10,
        batch_size=4,
        seed=0,
    )
    assert result.eval_length == 8 * L
    accs = [b.token_accuracy for b in result.buckets]
    assert accs == [1.0, 0.0, 0.0, 0.0]
    assert [b.sequence_accuracy for b in result.buckets] == [1.0, 0.0, 0.0, 0.0]
    assert [b.last_position_accuracy for b in result.buckets] == [1.0, 0.0, 0.0, 0.0]
    assert [b.num_scored for b in result.buckets] == [10 * L, 10 * L, 10 * 2 * L, 10 * 4 * L]
    assert [(b.start, b.end) for b in result.buckets] == [(0, L), (L, 2 * L), (2 * L, 4 * L), (4 * L, 8 * L)]
    assert [b.last_position_count for b in result.buckets] == [10, 10, 10, 10]


def test_wrong_only_at_position_L() -> None:
    task = get_task("z3")
    result = evaluate(
        _controlled_predictor(task, lambda p: p == L),
        task,
        train_length=L,
        num_sequences=6,
        batch_size=4,
        seed=1,
    )
    b1, b2, b3, b4 = result.buckets
    assert b1.token_accuracy == pytest.approx((L - 1) / L)
    assert b1.sequence_accuracy == 0.0
    assert b1.last_position_accuracy == 0.0
    for b in (b2, b3, b4):
        assert b.token_accuracy == 1.0
        assert b.sequence_accuracy == 1.0
        assert b.last_position_accuracy == 1.0


def test_wrong_only_at_position_L_plus_1() -> None:
    task = get_task("parity")
    result = evaluate(
        _controlled_predictor(task, lambda p: p == L + 1),
        task,
        train_length=L,
        num_sequences=6,
        batch_size=4,
        seed=2,
    )
    b1, b2, b3, b4 = result.buckets
    assert b1.token_accuracy == 1.0
    assert b1.sequence_accuracy == 1.0
    assert b2.token_accuracy == pytest.approx((L - 1) / L)
    assert b2.sequence_accuracy == 0.0
    assert b2.last_position_accuracy == 1.0
    assert (b3.token_accuracy, b4.token_accuracy) == (1.0, 1.0)


def test_wrong_only_at_each_last_position() -> None:
    task = get_task("z3")
    lasts = {L, 2 * L, 4 * L, 8 * L}
    result = evaluate(
        _controlled_predictor(task, lambda p: p in lasts),
        task,
        train_length=L,
        num_sequences=5,
        batch_size=2,
        seed=3,
    )
    assert [b.last_position_accuracy for b in result.buckets] == [0.0, 0.0, 0.0, 0.0]
    widths = [L, L, 2 * L, 4 * L]
    for b, w in zip(result.buckets, widths, strict=True):
        assert b.token_accuracy == pytest.approx((w - 1) / w)


class _EvenMaskedParity(Task):
    """Parity, but only odd 1-based positions are scored."""

    def __init__(self) -> None:
        self._base = get_task("parity")

    @property
    def name(self) -> str:
        return "parity_oddmask"

    @property
    def vocab_size(self) -> int:
        return 2

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def chance_accuracy(self) -> float:
        return 0.5

    def sample_inputs(self, batch: int, length: int, rng: np.random.Generator) -> IntArray:
        return self._base.sample_inputs(batch, length, rng)

    def labels(self, inputs: IntArray) -> IntArray:
        return self._base.labels(inputs)

    def mask(self, inputs: IntArray) -> BoolArray:
        m = np.zeros(inputs.shape, dtype=np.bool_)
        m[:, 0::2] = True  # 0-based even = 1-based odd
        return m


def test_evaluate_ignores_masked_out_positions() -> None:
    task = _EvenMaskedParity()
    # Wrong at every even 1-based position; those are all masked out.
    result = evaluate(
        _controlled_predictor(task, lambda p: p % 2 == 0),
        task,
        train_length=L,
        num_sequences=4,
        batch_size=4,
        seed=0,
    )
    for b, w in zip(result.buckets, [L, L, 2 * L, 4 * L], strict=True):
        assert b.token_accuracy == 1.0
        assert b.sequence_accuracy == 1.0
        assert b.num_scored == 4 * w // 2
        # Last positions (L, 2L, ...) are even, hence unscored.
        assert b.last_position_count == 0
        assert b.last_position_accuracy is None


def _flipflop_predictor(task: Task, wrong_on_reads: bool) -> Predictor:
    """True label on one token class, the wrong label on the other.

    wrong_on_reads=False: right on reads, wrong on writes and ignores.
    wrong_on_reads=True: wrong on reads, right on writes and ignores.
    """

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.cpu().numpy().astype(np.int64)
        labels = task.labels(x)
        is_read = x == FLIPFLOP_READ
        wrong = is_read if wrong_on_reads else ~is_read
        preds = np.where(wrong, 1 - labels, labels)
        return torch.nn.functional.one_hot(torch.from_numpy(preds), 2).to(torch.float32)

    return predict


@pytest.mark.parametrize("name", ["flipflop", "flipflop_sparse"])
def test_flipflop_scores_reads_only(name: str) -> None:
    task = get_task(name)
    n = 20
    data = task.sample(n, 8 * L, make_rng(name, "eval", 5))
    right = evaluate(_flipflop_predictor(task, False), task, train_length=L, num_sequences=n, batch_size=7, seed=5)
    wrong = evaluate(_flipflop_predictor(task, True), task, train_length=L, num_sequences=n, batch_size=7, seed=5)
    edges = (0, L, 2 * L, 4 * L, 8 * L)
    for k, (br, bw) in enumerate(zip(right.buckets, wrong.buckets, strict=True)):
        reads = int((data.inputs[:, edges[k] : edges[k + 1]] == FLIPFLOP_READ).sum())
        last_reads = int((data.inputs[:, edges[k + 1] - 1] == FLIPFLOP_READ).sum())
        assert br.num_scored == reads
        assert br.last_position_count == last_reads
        assert br.token_accuracy == 1.0
        assert bw.token_accuracy == 0.0
        if last_reads:
            assert br.last_position_accuracy == 1.0
            assert bw.last_position_accuracy == 0.0


def test_evaluate_rejects_bad_logit_shape() -> None:
    task = get_task("s3")

    def bad(inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(inputs.shape[0], inputs.shape[1], 5)

    with pytest.raises(ValueError):
        evaluate(bad, task, train_length=4, num_sequences=2, batch_size=2, seed=0)


def test_evaluate_rejects_non_finite_logits() -> None:
    task = get_task("parity")

    def nan(inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((inputs.shape[0], inputs.shape[1], 2), float("nan"))

    with pytest.raises(ValueError):
        evaluate(nan, task, train_length=4, num_sequences=2, batch_size=2, seed=0)


def test_evaluate_independent_of_batch_size() -> None:
    task = get_task("flipflop")
    rng_pred = np.random.default_rng(0)
    wrong_positions = set(int(p) for p in rng_pred.integers(1, 8 * L + 1, size=10))
    pred = _controlled_predictor(task, lambda p: p in wrong_positions)
    r1 = evaluate(pred, task, train_length=L, num_sequences=7, batch_size=7, seed=4)
    r2 = evaluate(pred, task, train_length=L, num_sequences=7, batch_size=3, seed=4)
    assert r1 == r2


def test_evaluate_uses_eval_stream() -> None:
    task = get_task("s5")
    seen: list[torch.Tensor] = []

    def record(inputs: torch.Tensor) -> torch.Tensor:
        seen.append(inputs.clone())
        return torch.zeros(inputs.shape[0], inputs.shape[1], task.num_classes)

    evaluate(record, task, train_length=4, num_sequences=3, batch_size=3, seed=9)
    expected = task.sample(3, 32, make_rng("s5", "eval", 9)).inputs
    np.testing.assert_array_equal(seen[0].numpy(), expected)


def test_result_json_round_trip() -> None:
    task = get_task("parity")
    result = evaluate(
        _controlled_predictor(task, lambda p: p > L),
        task,
        train_length=L,
        num_sequences=3,
        batch_size=2,
        seed=0,
    )
    assert isinstance(result, EvalResult)
    d = result.to_dict()
    text = json.dumps(d)
    back = json.loads(text)
    assert back["task"] == "parity"
    assert back["train_length"] == L
    assert back["buckets"][0]["token_accuracy"] == 1.0
    assert back["buckets"][1]["last_position"] == 2 * L
    with pytest.raises(AttributeError):
        result.seed = 3  # type: ignore[misc]


# -------------------------------------------------------------------- make_rng


def test_make_rng_deterministic() -> None:
    a = make_rng("s3", "train", 0).integers(0, 2**32, size=8)
    b = make_rng("s3", "train", 0).integers(0, 2**32, size=8)
    np.testing.assert_array_equal(a, b)


def test_make_rng_streams_differ() -> None:
    def draw(task: str, split: Split, seed: int) -> npt.NDArray[np.int64]:
        return make_rng(task, split, seed).integers(0, 2**62, size=8)

    base = draw("s3", "train", 0)
    for other in [draw("s3", "eval", 0), draw("s3", "val", 0), draw("s3", "train", 1), draw("s5", "train", 0)]:
        assert not np.array_equal(base, other)


def test_make_rng_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        make_rng("s3", "test", 0)  # type: ignore[arg-type]


def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        make_rng("s3", "train", -1)


# ------------------------------------------------- splits and errors for training


def test_evaluate_val_split_uses_val_stream() -> None:
    task = get_task("s5")
    seen: list[torch.Tensor] = []

    def record(inputs: torch.Tensor) -> torch.Tensor:
        seen.append(inputs.clone())
        return torch.zeros(inputs.shape[0], inputs.shape[1], task.num_classes)

    result = evaluate(record, task, train_length=4, num_sequences=3, batch_size=3, seed=9, split="val")
    expected = task.sample(3, 32, make_rng("s5", "val", 9)).inputs
    np.testing.assert_array_equal(seen[0].numpy(), expected)
    assert result.split == "val"
    assert result.to_dict()["split"] == "val"


def test_evaluate_default_split_is_eval() -> None:
    task = get_task("parity")
    result = evaluate(_controlled_predictor(task, lambda p: False), task, train_length=4, num_sequences=2)
    assert result.split == "eval"


def test_non_finite_logits_raise_dedicated_error() -> None:
    task = get_task("parity")

    def inf(inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((inputs.shape[0], inputs.shape[1], 2), float("inf"))

    with pytest.raises(NonFiniteLogitsError):
        evaluate(inf, task, train_length=4, num_sequences=2, batch_size=2, seed=0)


def test_eval_result_from_dict_round_trip() -> None:
    task = get_task("flipflop")
    result = evaluate(
        _controlled_predictor(task, lambda p: p % 7 == 0),
        task,
        train_length=L,
        num_sequences=5,
        batch_size=2,
        seed=1,
        split="val",
    )
    back = EvalResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert back == result
