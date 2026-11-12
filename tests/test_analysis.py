"""LR selection, per-seed bars, cell categories and aggregation.

The study's conclusions come straight out of this code, so the edge cases are
spelled out: ties, diverged runs, missing (None) buckets, values exactly at a
threshold, and seed counts other than 5.
"""

import pytest

from sst.analysis import (
    Bars,
    analyse,
    categorise,
    majority,
    seed_outcome,
    select_lr,
)
from tests.records import record

LRS = (3e-4, 1e-3, 3e-3)
FULL = (1.0, 1.0, 1.0, 1.0)


def _sweep(*vals: tuple[float | None, ...] | None, diverged: tuple[bool, ...] = (False, False, False)) -> list:  # type: ignore[type-arg]
    return [record(lr=lr, val=v, diverged=d) for lr, v, d in zip(LRS, vals, diverged, strict=True)]


# ------------------------------------------------------------------ select_lr


def test_selects_best_long_bucket_on_val() -> None:
    choice = select_lr(_sweep((1, 1, 1, 0.3), (1, 1, 1, 0.8), (1, 1, 1, 0.5)), LRS)
    assert choice is not None and choice.lr == 1e-3
    assert not choice.all_diverged


def test_long_bucket_tie_broken_by_higher_in_distribution_accuracy() -> None:
    choice = select_lr(_sweep((0.97, 1, 1, 0.5), (0.99, 1, 1, 0.5), (0.98, 1, 1, 0.5)), LRS)
    assert choice is not None and choice.lr == 1e-3


def test_full_tie_broken_by_lower_lr() -> None:
    choice = select_lr(_sweep((1, 1, 1, 0.5), (1, 1, 1, 0.5), (1, 1, 1, 0.5)), LRS)
    assert choice is not None and choice.lr == 3e-4
    choice = select_lr(_sweep((0.5, 1, 1, 0.2), (1, 1, 1, 0.5), (1, 1, 1, 0.5)), LRS)
    assert choice is not None and choice.lr == 1e-3


def test_long_bucket_dominates_in_distribution() -> None:
    choice = select_lr(_sweep((1.0, 1, 1, 0.50), (0.6, 1, 1, 0.51), (1.0, 1, 1, 0.5)), LRS)
    assert choice is not None and choice.lr == 1e-3


def test_selection_uses_val_split_not_eval() -> None:
    recs = [
        record(lr=3e-4, val=(1, 1, 1, 0.9), ev=(1, 1, 1, 0.1)),
        record(lr=1e-3, val=(1, 1, 1, 0.2), ev=(1, 1, 1, 1.0)),
        record(lr=3e-3, val=(1, 1, 1, 0.2), ev=(1, 1, 1, 1.0)),
    ]
    choice = select_lr(recs, LRS)
    assert choice is not None and choice.lr == 3e-4


def test_diverged_run_loses_to_any_finished_run() -> None:
    choice = select_lr(_sweep(None, (0.0, 0, 0, 0.0), None, diverged=(True, False, True)), LRS)
    assert choice is not None and choice.lr == 1e-3
    assert [c.status for c in choice.candidates] == ["diverged", "ok", "diverged"]


def test_none_long_bucket_ranks_below_zero() -> None:
    choice = select_lr(_sweep((1, 1, 1, None), (0.2, 1, 1, 0.0), (1, 1, 1, None)), LRS)
    assert choice is not None and choice.lr == 1e-3


def test_none_long_buckets_tie_then_in_distribution_decides() -> None:
    choice = select_lr(_sweep((0.5, 1, 1, None), (0.9, 1, 1, None), (None, 1, 1, None)), LRS)
    assert choice is not None and choice.lr == 1e-3


def test_all_diverged_picks_lowest_lr_and_says_so() -> None:
    choice = select_lr(_sweep(None, None, None, diverged=(True, True, True)), LRS)
    assert choice is not None and choice.lr == 3e-4 and choice.all_diverged


def test_incomplete_sweep_gives_none() -> None:
    assert select_lr(_sweep((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))[:2], LRS) is None
    assert select_lr([], LRS) is None


def test_ignores_other_seeds_and_off_grid_lrs() -> None:
    recs = _sweep((1, 1, 1, 0.2), (1, 1, 1, 0.3), (1, 1, 1, 0.4))
    recs.append(record(lr=1e-2, val=(1, 1, 1, 1.0)))
    recs.append(record(lr=3e-4, seed=1, val=(1, 1, 1, 1.0)))
    choice = select_lr(recs, LRS)
    assert choice is not None and choice.lr == 3e-3


def test_records_from_two_cells_rejected() -> None:
    recs = _sweep(FULL, FULL, FULL)
    recs[1] = record(lr=1e-3, depth=3)
    with pytest.raises(ValueError):
        select_lr(recs, LRS)


# ------------------------------------------------------------------ bars


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 4), (7, 4)])
def test_majority(n: int, expected: int) -> None:
    assert majority(n) == expected


def test_bars_inclusive_at_exact_threshold() -> None:
    o = seed_outcome(record(ev=(0.95, 0.5, 0.5, 0.90)), Bars())
    assert o.meets_id and o.meets_long and o.is_lg


@pytest.mark.parametrize(
    "ev,meets_id,is_lg",
    [
        ((0.9499999, 1.0, 1.0, 1.0), False, False),  # long fine, ID just short: not LG
        ((0.95, 1.0, 1.0, 0.8999999), True, False),
        ((None, 1.0, 1.0, 1.0), False, False),  # missing ID bucket fails the bar
        ((1.0, 1.0, 1.0, None), True, False),  # missing long bucket fails the bar
    ],
)
def test_bars_just_below_and_missing(ev: tuple[float | None, ...], meets_id: bool, is_lg: bool) -> None:
    o = seed_outcome(record(ev=ev), Bars())
    assert (o.meets_id, o.is_lg) == (meets_id, is_lg)


def test_bars_use_eval_split_not_val() -> None:
    o = seed_outcome(record(val=(0.0, 0, 0, 0.0), ev=FULL), Bars())
    assert o.is_lg


def test_diverged_seed_meets_no_bar() -> None:
    o = seed_outcome(record(diverged=True), Bars())
    assert o.status == "diverged"
    assert not o.meets_id and not o.meets_long and not o.is_lg
    assert o.bucket_accs == (None, None, None, None)


# ------------------------------------------------------------------ categories

LG = (1.0, 1.0, 1.0, 0.95)
ID = (0.99, 0.6, 0.4, 0.3)
FAIL = (0.5, 0.4, 0.3, 0.2)


def _outcomes(*evs: tuple[float | None, ...] | str) -> list:  # type: ignore[type-arg]
    out = []
    for seed, ev in enumerate(evs):
        if ev == "div":
            out.append(seed_outcome(record(seed=seed, diverged=True), Bars()))
        else:
            assert not isinstance(ev, str)
            out.append(seed_outcome(record(seed=seed, ev=ev), Bars()))
    return out


@pytest.mark.parametrize(
    "evs,category",
    [
        ((LG, LG, LG, LG, LG), "LG"),
        ((LG, LG, LG, FAIL, FAIL), "LG"),
        ((LG, LG, LG, "div", "div"), "LG"),
        ((LG, LG, FAIL, FAIL, FAIL), "LG-some"),
        ((LG, LG, "div", "div", "div"), "LG-some"),  # diverged seeds still count in the denominator
        ((LG, ID, ID, ID, ID), "LG-some"),
        ((LG, FAIL, FAIL, FAIL, FAIL), "LG-some"),
        ((ID, ID, ID, FAIL, FAIL), "ID-only"),
        ((ID, ID, ID, ID, ID), "ID-only"),
        ((ID, ID, FAIL, FAIL, FAIL), "F"),
        ((ID, ID, "div", "div", "div"), "F"),
        ((FAIL, FAIL, FAIL, FAIL, FAIL), "F"),
        (("div", "div", "div", "div", "div"), "F"),
    ],
)
def test_category_with_five_seeds(evs: tuple[tuple[float | None, ...] | str, ...], category: str) -> None:
    assert categorise(_outcomes(*evs), n_seeds=5) == category


@pytest.mark.parametrize(
    "evs,category",
    [
        ((LG, LG, FAIL), "LG"),
        ((LG, FAIL, FAIL), "LG-some"),
        ((ID, ID, FAIL), "ID-only"),
        ((ID, FAIL, FAIL), "F"),
    ],
)
def test_category_with_three_seeds(evs: tuple[tuple[float | None, ...], ...], category: str) -> None:
    assert categorise(_outcomes(*evs), n_seeds=3) == category


def test_category_incomplete_until_all_seeds_finish() -> None:
    assert categorise(_outcomes(LG, LG, LG, LG), n_seeds=5) == "incomplete"


def test_category_rejects_more_outcomes_than_seeds() -> None:
    with pytest.raises(ValueError):
        categorise(_outcomes(LG, LG, LG), n_seeds=2)


def test_category_rejects_duplicate_seeds() -> None:
    o = _outcomes(LG, LG, LG)
    with pytest.raises(ValueError):
        categorise([o[0], o[0], o[1]], n_seeds=3)


# ------------------------------------------------------------------ analyse


def _cell_records(
    chosen_evs: list[tuple[float | None, ...] | str], extra: bool = False
) -> list:  # type: ignore[type-arg]
    # Sweep: 1e-3 wins on val. Seeds 1.. run at 1e-3.
    recs = [
        record(lr=3e-4, val=(1, 1, 1, 0.1), ev=FAIL),
        record(lr=3e-3, val=(1, 1, 1, 0.2), ev=FAIL),
    ]
    for seed, ev in enumerate(chosen_evs):
        if ev == "div":
            recs.append(record(lr=1e-3, seed=seed, diverged=True))
        else:
            assert not isinstance(ev, str)
            recs.append(record(lr=1e-3, seed=seed, val=(1, 1, 1, 0.9), ev=ev))
    if extra:
        recs.append(record(lr=1e-3, seed=7, ev=FAIL))
        recs.append(record(lr=3e-4, seed=1, ev=LG))
    return recs


def test_analyse_aggregates_seeds_at_the_chosen_lr() -> None:
    evs: list[tuple[float | None, ...] | str] = [
        (1.0, 0.9, 0.8, 0.95),
        (0.96, 0.8, 0.7, 0.91),
        (1.0, 1.0, 0.9, None),
        "div",
        (0.5, 0.5, 0.4, 0.2),
    ]
    [cell] = analyse(_cell_records(evs, extra=True), LRS, n_seeds=5)
    assert cell.family == "delta" and cell.task == "s5" and cell.depth == 2
    assert cell.lr == 1e-3
    assert [s.seed for s in cell.seeds] == [0, 1, 2, 3, 4]
    assert cell.n_lg == 2 and cell.n_id == 3 and cell.n_long == 2
    assert cell.diverged_seeds == (3,)
    assert cell.category == "LG-some"
    first, last = cell.buckets[0], cell.buckets[-1]
    assert (first.start, first.end) == (0, 64) and (last.start, last.end) == (256, 512)
    assert first.n == 4 and first.median == pytest.approx(0.98) and first.min == 0.5 and first.max == 1.0
    assert last.n == 3 and last.median == pytest.approx(0.91) and last.min == 0.2 and last.max == 0.95


def test_analyse_marks_incomplete_cells() -> None:
    [cell] = analyse(_cell_records([LG, LG]), LRS, n_seeds=5)
    assert cell.category == "incomplete"
    assert cell.missing_seeds == (2, 3, 4)


def test_analyse_cell_with_incomplete_sweep() -> None:
    [cell] = analyse([record(lr=3e-4)], LRS, n_seeds=5)
    assert cell.lr is None and cell.category == "incomplete"


def test_analyse_all_diverged_bucket_stats_are_none() -> None:
    # Every sweep LR diverges, so the lowest LR is chosen; its seeds diverge too.
    recs = [record(lr=lr, diverged=True) for lr in LRS]
    recs += [record(lr=3e-4, seed=s, diverged=True) for s in (1, 2)]
    [cell] = analyse(recs, LRS, n_seeds=3)
    assert cell.lr == 3e-4
    assert cell.lr_choice is not None and cell.lr_choice.all_diverged
    assert cell.category == "F"
    assert cell.diverged_seeds == (0, 1, 2)
    assert all(b.n == 0 and b.median is None for b in cell.buckets)


def test_analyse_separates_cells() -> None:
    recs = _cell_records([LG, LG, LG]) + [record(lr=lr, depth=4) for lr in LRS]
    cells = analyse(recs, LRS, n_seeds=3)
    assert sorted(c.depth for c in cells) == [2, 4]
