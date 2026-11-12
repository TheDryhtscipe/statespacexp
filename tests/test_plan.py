"""Sweep plans, seed plans after a sweep, and family labels."""

import dataclasses

import pytest

from sst.config import RunConfig, spec_label
from sst.models import DeltaProductOptions, ModelSpec, TTTOptions
from sst.plan import LR_GRID, parse_family, seed_plan, sweep_plan
from tests.records import record

TEMPLATE = RunConfig(task="parity", model=ModelSpec("lstm"), lr=1.0, seed=0, steps=5000, batch_size=256)


def test_lr_grid_is_the_preregistered_one() -> None:
    assert LR_GRID == (3e-4, 1e-3, 3e-3)


def test_sweep_plan_crosses_families_tasks_depths_and_lrs() -> None:
    fams = [ModelSpec("delta"), ModelSpec("deltaproduct", options=DeltaProductOptions(n_householder=4))]
    plan = sweep_plan(fams, ["s5", "flipflop"], [1, 2, 4], TEMPLATE)
    assert len(plan) == 2 * 2 * 3 * 3
    assert len({c.run_id for c in plan}) == len(plan)
    assert {c.seed for c in plan} == {0}
    assert {c.lr for c in plan} == set(LR_GRID)
    assert {c.model.n_layers for c in plan} == {1, 2, 4}
    assert all(c.steps == 5000 and c.batch_size == 256 for c in plan)
    dp = [c for c in plan if c.model.mixer == "deltaproduct"]
    assert all(c.model.resolved_options() == DeltaProductOptions(n_householder=4) for c in dp)


@pytest.mark.parametrize(
    "spec",
    [
        ModelSpec("lstm"),
        ModelSpec("deltaproduct", options=DeltaProductOptions(n_householder=4)),
        ModelSpec("ttt", options=TTTOptions(inner_model="mlp", minibatch=1)),
        ModelSpec("ttt", options=TTTOptions(minibatch=16, eta_max=0.5)),
        ModelSpec("delta", d_model=64, use_short_conv=False),
    ],
)
def test_parse_family_inverts_spec_label(spec: ModelSpec) -> None:
    assert parse_family(spec_label(spec)) == dataclasses.replace(spec, options=spec.resolved_options())


def test_parse_family_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        parse_family("delta[nope=1]")


def test_seed_plan_waits_for_complete_sweep() -> None:
    recs = [record(lr=3e-4), record(lr=1e-3)]
    assert seed_plan(recs, n_seeds=5) == []


def test_seed_plan_emits_remaining_seeds_at_chosen_lr() -> None:
    recs = [
        record(lr=3e-4, val=(1, 1, 1, 0.2)),
        record(lr=1e-3, val=(1, 1, 1, 0.7)),
        record(lr=3e-3, val=(1, 1, 1, 0.4)),
    ]
    plan = seed_plan(recs, n_seeds=5)
    assert [c.seed for c in plan] == [1, 2, 3, 4]
    assert all(c.lr == 1e-3 for c in plan)
    base = recs[1].config
    assert all(dataclasses.replace(c, seed=0) == base for c in plan)


def test_seed_plan_covers_each_cell_separately() -> None:
    recs = [record(lr=lr, depth=d, val=(1, 1, 1, 0.5)) for lr in LR_GRID for d in (1, 2)]
    recs.append(record(lr=3e-4, depth=3))
    plan = seed_plan(recs, n_seeds=3)
    assert sorted((c.model.n_layers, c.seed) for c in plan) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(c.lr == 3e-4 for c in plan)  # full ties pick the lowest lr


def test_graphs_auto_turns_on_graphs_for_checked_mixers() -> None:
    from sst.plan import GRAPH_SAFE_MIXERS, _with_graphs

    plan = sweep_plan([ModelSpec("delta"), ModelSpec("lstm")], ["s5"], [2], TEMPLATE)
    assert all(not c.cuda_graphs for c in plan)
    auto = _with_graphs(plan, "auto")
    assert all(c.cuda_graphs == (c.model.mixer in GRAPH_SAFE_MIXERS) for c in auto)
    assert all((a.run_id == p.run_id) == (a.cuda_graphs == p.cuda_graphs) for a, p in zip(auto, plan, strict=True))
    assert all(not c.cuda_graphs for c in _with_graphs(auto, "off"))
