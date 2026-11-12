"""Sweep and seed plans.

The sweep for a cell is seed 0 at every LR of :data:`LR_GRID`. When a cell's
sweep is complete, :func:`seed_plan` chooses its LR (:func:`sst.analysis.select_lr`)
and emits seeds 1 .. n_seeds-1 at that LR, copying everything else from the
sweep run at that LR. Plans are lists of configs; the queue drops runs that are
recorded or already queued, so re-running ``advance`` is always safe.

Families are written as labels (:func:`sst.config.spec_label`), for example
``delta``, ``deltaproduct[n_householder=4]`` or ``ttt[inner_model=mlp]``.

Command line::

    python -m sst.plan sweep --families delta,lstm --tasks s5,flipflop --depths 1,2 \\
        --steps 20000 [--batch-size 128] [--priority 50] [--dry-run | --out plan.json]
    python -m sst.plan advance --n-seeds 5 [--priority 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from sst.analysis import group_by_cell, select_lr
from sst.config import RunConfig, ScalarField, config_to_dict, declared_scalar, replace_fields
from sst.evaluation import JSONValue
from sst.models import MIXER_KINDS, ModelSpec
from sst.results import RunResult, default_results_dir, load_runs

LR_GRID: tuple[float, ...] = (3e-4, 1e-3, 3e-3)

#: Mixers whose graphed training step matched eager within tolerance in the
#: throughput study (``python -m sst.bench table``, 20 steps at batch 128, s5,
#: lr 3e-3): every mixer kind, including deltaproduct n_h=4, TTT minibatch 16 and
#: TTT-MLP. ``--graphs auto`` turns graphs on for these. Re-run the check after
#: changing a mixer.
GRAPH_SAFE_MIXERS: frozenset[str] = frozenset(MIXER_KINDS)


def _coerce(text: str, field: ScalarField | None, name: str) -> bool | int | float | str | None:
    """Parse one ``name=text`` item of a family label as the field's declared type."""
    if field is None:
        raise ValueError(f"{name}: not a scalar field")
    if text == "None":
        if field.optional:
            return None
        raise ValueError(f"{name}: expected {field.kind.__name__}, got None")
    if field.kind is bool:
        if text in ("True", "true", "1"):
            return True
        if text in ("False", "false", "0"):
            return False
        raise ValueError(f"{name}: expected a bool, got {text!r}")
    try:
        if field.kind is int:
            return int(text)
        if field.kind is float:
            return float(text)
    except ValueError as exc:
        raise ValueError(f"{name}: cannot parse {text!r}") from exc
    if field.allowed is not None and text not in field.allowed:
        raise ValueError(f"{name}: expected one of {list(field.allowed)}, got {text!r}")
    return text


def parse_family(label: str) -> ModelSpec:
    """Inverse of :func:`sst.config.spec_label` (n_layers is left at its default)."""
    mixer, _, rest = label.partition("[")
    kind = next((k for k in MIXER_KINDS if k == mixer), None)
    if kind is None:
        raise ValueError(f"unknown mixer {mixer!r} in {label!r}")
    base = ModelSpec(kind)
    options = base.resolved_options()
    spec_changes: dict[str, JSONValue] = {}
    opt_changes: dict[str, JSONValue] = {}
    if rest:
        if not rest.endswith("]"):
            raise ValueError(f"malformed family label {label!r}")
        spec_fields = {f.name for f in dataclasses.fields(base)} - {"mixer", "options", "n_layers"}
        opt_fields = {f.name for f in dataclasses.fields(options)}
        for item in rest[:-1].split(","):
            key, eq, value = item.partition("=")
            if not eq:
                raise ValueError(f"malformed option {item!r} in {label!r}")
            if key in spec_fields:
                spec_changes[key] = _coerce(value, declared_scalar(type(base), key), key)
            elif key in opt_fields:
                opt_changes[key] = _coerce(value, declared_scalar(type(options), key), key)
            else:
                raise ValueError(f"unknown field {key!r} for {kind}")
    options = replace_fields(options, opt_changes)
    return dataclasses.replace(replace_fields(base, spec_changes), options=options)


def sweep_plan(
    families: Sequence[ModelSpec],
    tasks: Sequence[str],
    depths: Sequence[int],
    template: RunConfig,
    lrs: Sequence[float] = LR_GRID,
    sweep_seed: int = 0,
) -> list[RunConfig]:
    """Seed ``sweep_seed`` at every LR for every family x task x depth; other fields from ``template``."""
    out: list[RunConfig] = []
    for spec in families:
        for task in tasks:
            for depth in depths:
                for lr in lrs:
                    out.append(dataclasses.replace(
                        template,
                        task=task,
                        model=dataclasses.replace(spec, n_layers=depth),
                        lr=lr,
                        seed=sweep_seed,
                    ))
    return out


def seed_plan(records: Sequence[RunResult], n_seeds: int, lrs: Sequence[float] = LR_GRID,
              sweep_seed: int = 0) -> list[RunConfig]:
    """Seed runs for every cell whose sweep is complete (see the module docstring)."""
    out: list[RunConfig] = []
    for cell in group_by_cell(records).values():
        choice = select_lr(cell, lrs, sweep_seed)
        if choice is None:
            continue
        base = next(r.config for r in cell if r.run_id == next(
            c.run_id for c in choice.candidates if c.lr == choice.lr))
        out.extend(dataclasses.replace(base, seed=s) for s in range(n_seeds) if s != sweep_seed)
    return out


def _with_graphs(configs: Sequence[RunConfig], mode: str) -> list[RunConfig]:
    if mode == "keep":
        return list(configs)
    return [
        dataclasses.replace(c, cuda_graphs=(mode == "on" or (mode == "auto" and c.model.mixer in GRAPH_SAFE_MIXERS)))
        for c in configs
    ]


def main(argv: Sequence[str] | None = None) -> int:
    from sst.queue import DEFAULT_PRIORITY, JobQueue

    parser = argparse.ArgumentParser(prog="python -m sst.plan", description="Build and submit sweep and seed plans")
    parser.add_argument("--results", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="LR sweep (seed 0) for families x tasks x depths")
    s.add_argument("--families", required=True, help="comma-separated family labels")
    s.add_argument("--tasks", required=True)
    s.add_argument("--depths", required=True)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--d-model", type=int, default=None)
    s.add_argument("--graphs", choices=("auto", "on", "off"), default="auto")
    a = sub.add_parser("advance", help="seed runs for cells whose sweep is complete")
    a.add_argument("--n-seeds", type=int, default=5)
    a.add_argument("--graphs", choices=("auto", "on", "off", "keep"), default="keep")
    for p in (s, a):
        p.add_argument("--priority", type=int, default=DEFAULT_PRIORITY)
        p.add_argument("--note", default="")
        p.add_argument("--dry-run", action="store_true", help="print the plan, do not submit")
        p.add_argument("--out", type=Path, default=None, help="write the plan to a file instead of submitting")
    args = parser.parse_args(argv)
    results_dir: Path = args.results if args.results is not None else default_results_dir()

    if args.cmd == "sweep":
        families = [parse_family(f) for f in args.families.split(",")]
        if args.d_model is not None:
            families = [dataclasses.replace(f, d_model=args.d_model) for f in families]
        template = RunConfig(task="parity", model=ModelSpec("lstm"), lr=1.0, seed=0, steps=args.steps,
                             batch_size=args.batch_size)
        plan = sweep_plan(families, args.tasks.split(","), [int(d) for d in args.depths.split(",")], template)
        note = args.note or "sweep"
    else:
        plan = seed_plan(load_runs(results_dir / "runs.jsonl"), args.n_seeds)
        note = args.note or "seeds"
    plan = _with_graphs(plan, args.graphs)

    if args.out is not None:
        args.out.write_text(json.dumps([config_to_dict(c) for c in plan], indent=1))
        print(f"wrote {len(plan)} configs to {args.out}")
    elif args.dry_run:
        for c in plan:
            print(c.run_id, c.task, c.model.mixer, f"d={c.model.n_layers}", f"lr={c.lr:g}", f"seed={c.seed}",
                  f"graphs={c.cuda_graphs}")
        print(f"{len(plan)} configs (dry run)")
    else:
        print(JobQueue(results_dir).submit(plan, priority=args.priority, note=note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
