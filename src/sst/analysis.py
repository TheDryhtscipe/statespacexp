"""LR selection, per-seed bars, cell categories and per-cell aggregation.

Cells
-----
A cell is one (family, task, depth) under one protocol: all runs whose configs
agree on everything except ``lr`` and ``seed`` (:attr:`RunConfig.cell_key`).

LR selection (:func:`select_lr`)
--------------------------------
Once seed ``sweep_seed`` (0) has a record at every LR of the grid, the chosen LR
maximises, in order:

1. token accuracy in the last bucket ((4L, 8L], i.e. (256, 512] at L=64) on the
   **val** split;
2. token accuracy in the first bucket ([1, L], in-distribution) on the val split;
3. lower LR.

A diverged run, and a missing (``None``) bucket, rank below every number,
including 0.0. If every LR diverged the lowest LR is chosen and
``all_diverged`` is set.

Seed bars and categories (:func:`seed_outcome`, :func:`categorise`)
-------------------------------------------------------------------
Bars use the **eval** split, inclusive (``>=``):

* a seed *meets ID* if token accuracy in [1, L] >= 0.95;
* a seed *meets long* if token accuracy in (4L, 8L] >= 0.90;
* a seed is *LG* if it meets both.

A diverged seed, or a ``None`` bucket, meets no bar. Diverged seeds stay in the
denominator: they are seeds that failed, not seeds that are missing.

With ``n_seeds`` seeds (0 .. n_seeds-1 at the chosen LR) and
``m = majority(n_seeds) = n_seeds // 2 + 1`` (a strict majority):

* **LG**       at least m seeds are LG;
* **LG-some**  1 to m-1 seeds are LG;
* **ID-only**  no seed is LG, and at least m seeds meet ID;
* **F**        otherwise;
* **incomplete** while fewer than n_seeds seeds have a record.

With the pre-registered 5 seeds, m = 3, which gives the protocol's "at least 3
seeds" (LG, ID-only) and "1-2 seeds" (LG-some). Writing the rule as a majority
keeps it coherent if the seed count changes: 3 seeds give m = 2, 7 give m = 4.

Aggregation
-----------
Per bucket, the median, min and max of eval token accuracy over the seeds that
finished (status ok) and have a value there; ``n`` says how many that was.
Diverged seeds are listed in ``diverged_seeds``, not folded into the statistics.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sst.config import family_key, spec_label
from sst.evaluation import EvalResult
from sst.results import RunResult, Status

Category = Literal["LG", "LG-some", "ID-only", "F", "incomplete"]
CATEGORIES: tuple[Category, ...] = ("LG", "LG-some", "ID-only", "F", "incomplete")


@dataclass(frozen=True)
class Bars:
    id_bar: float = 0.95
    long_bar: float = 0.90


def majority(n_seeds: int) -> int:
    """Smallest strict majority of ``n_seeds``."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    return n_seeds // 2 + 1


def _lr_equal(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def _rank(x: float | None) -> float:
    return -math.inf if x is None else x


def _first_last(result: EvalResult | None) -> tuple[float | None, float | None]:
    if result is None or not result.buckets:
        return None, None
    return result.buckets[0].token_accuracy, result.buckets[-1].token_accuracy


# ------------------------------------------------------------------ LR selection


@dataclass(frozen=True)
class LRCandidate:
    lr: float
    run_id: str
    status: Status
    val_long: float | None
    val_id: float | None


@dataclass(frozen=True)
class LRChoice:
    lr: float
    all_diverged: bool
    candidates: tuple[LRCandidate, ...]


def select_lr(records: Sequence[RunResult], lrs: Sequence[float], sweep_seed: int = 0) -> LRChoice | None:
    """Choose a cell's LR from its sweep; ``None`` until every LR in ``lrs`` has a record.

    Records with another seed or an LR outside ``lrs`` are ignored. Raises
    ``ValueError`` if the sweep records come from more than one cell.
    """
    sweep = [r for r in records if r.config.seed == sweep_seed and any(_lr_equal(r.config.lr, lr) for lr in lrs)]
    if len({r.config.cell_key for r in sweep}) > 1:
        raise ValueError("sweep records come from more than one cell")
    candidates: list[LRCandidate] = []
    for lr in sorted(lrs):
        match = next((r for r in sweep if _lr_equal(r.config.lr, lr)), None)
        if match is None:
            return None
        val_id, val_long = (None, None) if match.status != "ok" else _first_last(match.val)
        candidates.append(LRCandidate(lr, match.run_id, match.status, val_long, val_id))
    best = max(candidates, key=lambda c: (_rank(c.val_long), _rank(c.val_id), -c.lr))
    return LRChoice(best.lr, all(c.status != "ok" for c in candidates), tuple(candidates))


# ------------------------------------------------------------------ bars and categories


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    run_id: str
    status: Status
    bucket_accs: tuple[float | None, ...]
    meets_id: bool
    meets_long: bool
    is_lg: bool


def seed_outcome(r: RunResult, bars: Bars, n_buckets: int = 4) -> SeedOutcome:
    if r.status != "ok" or r.eval is None:
        return SeedOutcome(r.config.seed, r.run_id, r.status, (None,) * n_buckets, False, False, False)
    accs = tuple(b.token_accuracy for b in r.eval.buckets)
    first, last = accs[0], accs[-1]
    meets_id = first is not None and first >= bars.id_bar
    meets_long = last is not None and last >= bars.long_bar
    return SeedOutcome(r.config.seed, r.run_id, r.status, accs, meets_id, meets_long, meets_id and meets_long)


def categorise(outcomes: Sequence[SeedOutcome], n_seeds: int) -> Category:
    """Category of a cell from its seeds' outcomes; see the module docstring."""
    seeds = [o.seed for o in outcomes]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {seeds}")
    if len(outcomes) > n_seeds:
        raise ValueError(f"{len(outcomes)} outcomes for {n_seeds} seeds")
    if len(outcomes) < n_seeds:
        return "incomplete"
    m = majority(n_seeds)
    n_lg = sum(o.is_lg for o in outcomes)
    n_id = sum(o.meets_id for o in outcomes)
    if n_lg >= m:
        return "LG"
    if n_lg >= 1:
        return "LG-some"
    if n_id >= m:
        return "ID-only"
    return "F"


# ------------------------------------------------------------------ aggregation


@dataclass(frozen=True)
class BucketStats:
    start: int
    end: int
    n: int
    median: float | None
    min: float | None
    max: float | None


@dataclass(frozen=True)
class CellSummary:
    family: str
    family_key: str
    task: str
    depth: int
    cell_key: str
    n_seeds: int
    lr: float | None
    lr_choice: LRChoice | None
    seeds: tuple[SeedOutcome, ...]
    missing_seeds: tuple[int, ...]
    n_lg: int
    n_id: int
    n_long: int
    diverged_seeds: tuple[int, ...]
    buckets: tuple[BucketStats, ...]
    category: Category


def _bucket_stats(outcomes: Sequence[SeedOutcome], edges: Sequence[tuple[int, int]]) -> tuple[BucketStats, ...]:
    out: list[BucketStats] = []
    for k, (start, end) in enumerate(edges):
        values = [v for o in outcomes if o.status == "ok" for v in [o.bucket_accs[k]] if v is not None]
        if values:
            out.append(BucketStats(start, end, len(values), statistics.median(values), min(values), max(values)))
        else:
            out.append(BucketStats(start, end, 0, None, None, None))
    return tuple(out)


def _edges(records: Sequence[RunResult]) -> list[tuple[int, int]]:
    for r in records:
        if r.eval is not None:
            return [(b.start, b.end) for b in r.eval.buckets]
    c = records[0].config
    edges = [0, c.train_length]
    while edges[-1] < c.eval_multiplier * c.train_length:
        edges.append(2 * edges[-1])
    return list(zip(edges[:-1], edges[1:], strict=True))


def summarise_cell(records: Sequence[RunResult], lrs: Sequence[float], n_seeds: int,
                   bars: Bars = Bars(), sweep_seed: int = 0) -> CellSummary:
    """Summarise one cell's records (all with the same cell key)."""
    if not records:
        raise ValueError("no records")
    if len({r.config.cell_key for r in records}) != 1:
        raise ValueError("records come from more than one cell")
    first: dict[str, RunResult] = {}
    for r in records:  # first record per run id wins
        first.setdefault(r.run_id, r)
    unique = list(first.values())
    cfg = unique[0].config
    edges = _edges(unique)
    choice = select_lr(unique, lrs, sweep_seed)
    outcomes: list[SeedOutcome] = []
    missing: list[int] = list(range(n_seeds))
    if choice is not None:
        by_seed = {r.config.seed: r for r in unique if _lr_equal(r.config.lr, choice.lr)}
        outcomes = [seed_outcome(by_seed[s], bars, len(edges)) for s in range(n_seeds) if s in by_seed]
        missing = [s for s in range(n_seeds) if s not in by_seed]
    category: Category = categorise(outcomes, n_seeds) if choice is not None else "incomplete"
    return CellSummary(
        family=spec_label(cfg.model),
        family_key=family_key(cfg.model),
        task=cfg.task,
        depth=cfg.model.n_layers,
        cell_key=cfg.cell_key,
        n_seeds=n_seeds,
        lr=None if choice is None else choice.lr,
        lr_choice=choice,
        seeds=tuple(outcomes),
        missing_seeds=tuple(missing),
        n_lg=sum(o.is_lg for o in outcomes),
        n_id=sum(o.meets_id for o in outcomes),
        n_long=sum(o.meets_long for o in outcomes),
        diverged_seeds=tuple(o.seed for o in outcomes if o.status != "ok"),
        buckets=_bucket_stats(outcomes, edges),
        category=category,
    )


def group_by_cell(records: Sequence[RunResult]) -> dict[str, list[RunResult]]:
    cells: dict[str, list[RunResult]] = defaultdict(list)
    for r in records:
        cells[r.config.cell_key].append(r)
    return dict(cells)


def analyse(records: Sequence[RunResult], lrs: Sequence[float], n_seeds: int,
            bars: Bars = Bars(), sweep_seed: int = 0) -> list[CellSummary]:
    """One summary per cell, sorted by family, task and depth."""
    out = [summarise_cell(rs, lrs, n_seeds, bars, sweep_seed) for rs in group_by_cell(records).values()]
    return sorted(out, key=lambda c: (c.family, c.task, c.depth, c.cell_key))


def format_table(cells: Sequence[CellSummary]) -> str:
    """A plain-text table: one row per cell, eval-split median [min, max] per bucket."""
    lines: list[str] = []
    for c in cells:
        buckets = "  ".join(
            f"({b.start},{b.end}] "
            + ("-" if b.median is None else f"{b.median:.3f} [{b.min:.3f},{b.max:.3f}] n={b.n}")
            for b in c.buckets
        )
        lr = "-" if c.lr is None else f"{c.lr:g}"
        div = f" diverged={list(c.diverged_seeds)}" if c.diverged_seeds else ""
        miss = f" missing={list(c.missing_seeds)}" if c.missing_seeds and c.lr is not None else ""
        flag = " ALL-LR-DIVERGED" if c.lr_choice is not None and c.lr_choice.all_diverged else ""
        lines.append(
            f"{c.family:<32} {c.task:<16} d={c.depth} lr={lr:<7} {c.category:<10} "
            f"LG={c.n_lg} ID={c.n_id} long={c.n_long}/{c.n_seeds}{div}{miss}{flag}  {buckets}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from sst.plan import LR_GRID
    from sst.results import default_results_dir, load_runs

    parser = argparse.ArgumentParser(prog="python -m sst.analysis", description="Per-cell categories")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args(argv)
    results_dir: Path = args.results if args.results is not None else default_results_dir()
    print(format_table(analyse(load_runs(results_dir / "runs.jsonl"), LR_GRID, args.n_seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
