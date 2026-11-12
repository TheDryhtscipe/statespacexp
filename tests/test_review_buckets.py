"""Bucket arithmetic against an independent loop, written in review (statespacexp-a574dd).

The predictor is right on an exact, random set of (sequence, position) pairs
and wrong everywhere else. Every reported number is recomputed here with
plain Python loops over 1-based positions.
"""

import numpy as np
import pytest
import torch

from sst.evaluation import evaluate, make_rng
from sst.tasks import get_task


@pytest.mark.parametrize(
    "name,L,n,batch_size", [("s5", 8, 37, 5), ("flipflop_sparse", 16, 50, 7), ("parity", 3, 11, 4)]
)
def test_every_reported_number_matches_own_count(name: str, L: int, n: int, batch_size: int) -> None:
    task = get_task(name)
    k, length = task.num_classes, 8 * L
    data = task.sample(n, length, make_rng(name, "eval", 3))
    chosen = np.random.default_rng(42).random((n, length)) < 0.7
    chosen[0, :] = True
    chosen[1, :] = False
    row_of = {row.tobytes(): i for i, row in enumerate(data.inputs)}
    assert len(row_of) == n

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        rows = [row_of[r.tobytes()] for r in inputs.numpy()]
        labels = data.targets[rows]
        cls = np.where(chosen[rows], labels, (labels + 1) % k)
        return torch.nn.functional.one_hot(torch.from_numpy(cls), k).float()

    result = evaluate(predict, task, train_length=L, num_sequences=n, batch_size=batch_size, seed=3)
    assert result.eval_length == length
    bounds = [(1, L), (L + 1, 2 * L), (2 * L + 1, 4 * L), (4 * L + 1, 8 * L)]
    assert len(result.buckets) == len(bounds)
    for bucket, (first, last) in zip(result.buckets, bounds, strict=True):
        tok_right = tok_n = seq_right = seq_n = last_right = last_n = 0
        for i in range(n):
            any_scored, all_right = False, True
            for p in range(first, last + 1):  # 1-based position p is column p - 1
                if data.mask[i, p - 1]:
                    tok_n += 1
                    tok_right += int(chosen[i, p - 1])
                    any_scored = True
                    all_right = all_right and bool(chosen[i, p - 1])
            if any_scored:
                seq_n += 1
                seq_right += int(all_right)
            if data.mask[i, last - 1]:
                last_n += 1
                last_right += int(chosen[i, last - 1])
        assert (bucket.start, bucket.end, bucket.last_position) == (first - 1, last, last)
        assert bucket.num_scored == tok_n
        assert bucket.token_accuracy == pytest.approx(tok_right / tok_n)
        assert bucket.num_sequences_scored == seq_n
        assert bucket.sequence_accuracy == pytest.approx(seq_right / seq_n)
        assert bucket.last_position_count == last_n
        if last_n:
            assert bucket.last_position_accuracy == pytest.approx(last_right / last_n)
        else:
            assert bucket.last_position_accuracy is None
