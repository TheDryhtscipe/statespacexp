"""Distribution and shortcut checks written in review (statespacexp-a574dd)."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from sst.evaluation import Split, evaluate, make_rng
from sst.tasks import get_task

GROUP_TASKS = ["parity", "z3", "s3", "s5"]
FLIPFLOP_TASKS = [("flipflop", 0.1), ("flipflop_sparse", 0.02)]


@pytest.mark.parametrize("name", GROUP_TASKS)
def test_group_labels_uniform_at_every_position(name: str) -> None:
    task = get_task(name)
    k = task.num_classes
    n = 4000 if k < 100 else 24000
    data = task.sample(n, 64, make_rng(name, "eval", 0))
    counts = np.stack([np.bincount(data.targets[:, t], minlength=k) for t in range(64)])
    expected = n / k
    chi2 = ((counts - expected) ** 2 / expected).sum(axis=1)
    # Mean chi-square over positions is close to its df under uniformity.
    assert abs(float(chi2.mean()) - (k - 1)) < 0.25 * (k - 1) + 1.0
    # No single position is far from uniform (generous bound: df + 8 sd).
    assert float(chi2.max()) < (k - 1) + 8 * np.sqrt(2 * (k - 1))


@pytest.mark.parametrize("name,p_write", FLIPFLOP_TASKS)
def test_flipflop_reads_and_gaps_match_p_write(name: str, p_write: float) -> None:
    data = get_task(name).sample(2000, 512, make_rng(name, "eval", 0))
    rest = data.inputs[:, 1:]
    assert float(np.isin(rest, [0, 1]).mean()) == pytest.approx(p_write, rel=0.05)
    assert float((rest == 2).mean()) == pytest.approx((1 - p_write) / 2, abs=0.005)
    # Gap from each read back to its last write. Away from the start, this is
    # geometric: P(gap > k) = (1 - p_write) ** k.
    pos = np.arange(512)
    is_write = np.isin(data.inputs, [0, 1])
    last_write = np.maximum.accumulate(np.where(is_write, pos, -1), axis=1)
    gaps = (pos - last_write)[:, 256:][data.mask[:, 256:]]
    for k in (5, 20, 50):
        assert float((gaps > k).mean()) == pytest.approx((1 - p_write) ** k, abs=0.02)
    assert float(data.targets[data.mask].mean()) == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("name", GROUP_TASKS + [n for n, _ in FLIPFLOP_TASKS])
def test_splits_differ_and_eval_is_reproducible(name: str) -> None:
    task = get_task(name)

    def draw(split: Split, seed: int) -> npt.NDArray[np.int64]:
        return task.sample(64, 128, make_rng(name, split, seed)).inputs

    ev = draw("eval", 0)
    np.testing.assert_array_equal(ev, draw("eval", 0))
    others = [draw("train", 0), draw("val", 0), draw("eval", 1)]
    for other in others:
        assert not np.array_equal(ev, other)
        # Not merely different: no eval sequence appears in the other set.
        seen = {row.tobytes() for row in other}
        assert not any(row.tobytes() in seen for row in ev)


def _lookup_predictor(table: npt.NDArray[np.int64], num_classes: int, use_token: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    def predict(inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.cpu().numpy()
        t = np.arange(x.shape[1])[None, :]
        cls = table[t, x] if use_token else np.broadcast_to(table[t, 0], x.shape)
        return torch.nn.functional.one_hot(torch.from_numpy(np.ascontiguousarray(cls)), num_classes).float()

    return predict


@pytest.mark.parametrize("name", GROUP_TASKS + [n for n, _ in FLIPFLOP_TASKS])
@pytest.mark.parametrize("use_token", [False, True])
def test_position_and_token_without_history_stay_at_chance(name: str, use_token: bool) -> None:
    """Best lookup from (position) or (position, current token), fitted on train data.

    It is fitted at the full eval length, so it has more information than a
    real run. Beyond position 1 it must not beat chance.
    """
    task = get_task(name)
    k, v, length, L = task.num_classes, task.vocab_size, 128, 16
    train = task.sample(6000, length, make_rng(name, "train", 0))
    counts = np.zeros((length, v, k), dtype=np.int64)
    for t in range(length):
        m = train.mask[:, t]
        tok = train.inputs[m, t] if use_token else np.zeros(int(m.sum()), dtype=np.int64)
        np.add.at(counts[t], (tok, train.targets[m, t]), 1)
    table = counts.argmax(axis=2).astype(np.int64)
    result = evaluate(
        _lookup_predictor(table, k, use_token), task, train_length=L, num_sequences=2000, batch_size=500, seed=0
    )
    chance = task.chance_accuracy
    tol = 0.03 if "flipflop" in name else max(0.01, 4 * np.sqrt(chance * (1 - chance) / (2000 * L)))
    for b in result.buckets[1:]:
        assert b.token_accuracy is not None
        assert b.token_accuracy == pytest.approx(chance, abs=tol)
