"""Independent label checks written in review (statespacexp-a574dd).

The permutations are enumerated here by recursion, not with itertools or
anything in ``sst``, and composed by hand under the convention documented in
``sst.groups``: s_t = s_{t-1} . g_t with (a . b)(i) = a(b(i)).
"""

import numpy as np
import pytest

from sst.evaluation import make_rng
from sst.tasks import get_task

Perm = tuple[int, ...]


def _lex_perms(items: list[int]) -> list[Perm]:
    if not items:
        return [()]
    out: list[Perm] = []
    for k, x in enumerate(items):
        for rest in _lex_perms(items[:k] + items[k + 1 :]):
            out.append((x, *rest))
    return out


@pytest.mark.parametrize("name,n,batch", [("s3", 3, 300), ("s5", 5, 100)])
def test_symmetric_labels_match_own_composition(name: str, n: int, batch: int) -> None:
    perms = _lex_perms(list(range(n)))
    index = {p: i for i, p in enumerate(perms)}
    assert index[tuple(range(n))] == 0
    data = get_task(name).sample(batch, 40, make_rng(name, "train", 0))
    opposite_mismatches = 0
    for row, labels in zip(data.inputs, data.targets, strict=True):
        state: Perm = perms[int(row[0])]
        other: Perm = state
        assert labels[0] == index[state]
        for t in range(1, len(row)):
            g = perms[int(row[t])]
            state = tuple(state[g[i]] for i in range(n))  # s_{t-1} . g_t
            other = tuple(g[other[i]] for i in range(n))  # g_t . s_{t-1}
            assert labels[t] == index[state]
            opposite_mismatches += int(labels[t] != index[other])
    # The opposite convention must disagree often, or this test proves nothing.
    assert opposite_mismatches > 0.5 * batch * 39


@pytest.mark.parametrize("name,n", [("parity", 2), ("z3", 3)])
def test_cyclic_labels_match_running_sum(name: str, n: int) -> None:
    data = get_task(name).sample(500, 100, make_rng(name, "train", 0))
    expected = np.empty_like(data.inputs)
    for b in range(data.inputs.shape[0]):
        total = 0
        for t in range(data.inputs.shape[1]):
            total = (total + int(data.inputs[b, t])) % n
            expected[b, t] = total
    np.testing.assert_array_equal(data.targets, expected)


@pytest.mark.parametrize("name", ["flipflop", "flipflop_sparse"])
def test_flipflop_labels_and_mask_match_own_scan(name: str) -> None:
    data = get_task(name).sample(200, 300, make_rng(name, "train", 0))
    for row, labels, mask in zip(data.inputs, data.targets, data.mask, strict=True):
        bit = -1
        for t, tok in enumerate(row.tolist()):
            if tok in (0, 1):
                bit = tok
            assert bit in (0, 1)
            assert mask[t] == (tok == 2)
            if mask[t]:
                assert labels[t] == bit
