"""Training throughput: steps per second per mixer, eager and with CUDA graphs.

Uses the training step of :mod:`sst.train` (same data stream, loss, clipping and
AdamW), so the numbers are what a run will see. Results go to stdout as a table
and, with ``--out``, as JSON lines.

Command line::

    python -m sst.bench table [--batches 128,256] [--seconds 3] [--out bench.jsonl]
    python -m sst.bench one --family delta --batch 128 --seconds 20 [--graphs] [--start-at T]
    python -m sst.bench concurrency --family delta --procs 1,2,3 [--graphs]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sst.config import RunConfig, spec_label
from sst.evaluation import JSONValue
from sst.models import ModelSpec, build_model
from sst.models.spec import MIXER_KINDS, DeltaProductOptions, TTTOptions
from sst.plan import parse_family
from sst.tasks import get_task
from sst.train import Stepper, TrainStream, graph_equivalence, lr_at

BENCH_TASK = "s5"


def bench_families() -> list[ModelSpec]:
    """Every mixer at defaults, plus deltaproduct n_h=4, TTT as published, and TTT-MLP at minibatch 1."""
    specs = [ModelSpec(k) for k in MIXER_KINDS]
    specs.append(ModelSpec("deltaproduct", options=DeltaProductOptions(n_householder=4)))
    specs.append(ModelSpec("ttt", options=TTTOptions(inner_model="linear", inner_norm=True, minibatch=16, eta_max=1.0)))
    specs.append(ModelSpec("ttt", options=TTTOptions(inner_model="mlp", minibatch=1)))
    return specs


@dataclass(frozen=True)
class Measurement:
    family: str
    batch: int
    graphs: bool
    steps_per_s: float | None
    steps_timed: int
    peak_mem_mb: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "family": self.family, "batch": self.batch, "graphs": self.graphs,
            "steps_per_s": self.steps_per_s, "steps_timed": self.steps_timed,
            "peak_mem_mb": self.peak_mem_mb, "error": self.error,
        }


def _config(spec: ModelSpec, batch: int, graphs: bool) -> RunConfig:
    return RunConfig(task=BENCH_TASK, model=spec, lr=1e-3, seed=0, steps=100_000, batch_size=batch,
                     warmup_steps=500, cuda_graphs=graphs)


def measure(spec: ModelSpec, batch: int, graphs: bool, seconds: float, device: str = "cuda",
            start_at: float | None = None) -> Measurement:
    """Time training steps for about ``seconds`` after warm-up (and graph capture)."""
    dev = torch.device(device)
    label = spec_label(spec)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        cfg = _config(spec, batch, graphs)
        task = get_task(cfg.task)
        torch.manual_seed(0)
        model = build_model(spec, task.vocab_size, task.num_classes).to(dev)
        stepper = Stepper(model, cfg, dev, graphs)
        stream = iter(TrainStream(task, cfg, dev))
        step = 0

        def run(n: int) -> None:
            nonlocal step
            for _ in range(n):
                x, y, m = next(stream)
                stepper.step(x, y, m, lr_at(step, peak=cfg.lr, warmup=cfg.warmup_steps, total=cfg.steps))
                step += 1

        run(6)  # eager warm-up, graph capture and first replays
        torch.cuda.synchronize(dev)
        if start_at is not None:
            time.sleep(max(0.0, start_at - time.time()))
        t0 = time.perf_counter()
        run(5)
        torch.cuda.synchronize(dev)
        per_step = (time.perf_counter() - t0) / 5
        n = max(10, int(seconds / max(per_step, 1e-4)))
        t0 = time.perf_counter()
        run(n)
        torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - t0
        _, bad = stepper.read_window()
        peak = torch.cuda.max_memory_allocated(dev) / 2**20
        del stepper, model
        return Measurement(label, batch, graphs, n / elapsed, n, peak, "non-finite loss" if bad else None)
    except torch.OutOfMemoryError:
        return Measurement(label, batch, graphs, None, 0, None, "out of memory")
    except RuntimeError as exc:
        return Measurement(label, batch, graphs, None, 0, None, f"{type(exc).__name__}: {str(exc)[:200]}")
    finally:
        torch.cuda.empty_cache()


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def table(batches: Sequence[int], seconds: float, out: str | None, families: Sequence[ModelSpec],
          equivalence: bool) -> None:
    rows: list[str] = []
    header = (f"{'family':<34}" + "".join(f"{'b' + str(b) + ' eager':>11}{'graphs':>9}{'x':>6}" for b in batches)
              + f"{'mem MB b' + str(batches[-1]):>13}" + ("  graph check (max rel diff / noise)" if equivalence else ""))
    print(header, flush=True)
    for spec in families:
        cells: list[str] = []
        mem: float | None = None
        for b in batches:
            eager = measure(spec, b, False, seconds)
            graph = measure(spec, b, True, seconds)
            for m in (eager, graph):
                if out is not None:
                    with open(out, "a") as f:
                        f.write(json.dumps(m.to_dict()) + "\n")
            speedup = (graph.steps_per_s / eager.steps_per_s
                       if graph.steps_per_s is not None and eager.steps_per_s else None)
            cells.append(f"{_fmt(eager.steps_per_s):>11}{_fmt(graph.steps_per_s):>9}"
                         f"{'-' if speedup is None else f'{speedup:.1f}':>6}")
            mem = graph.peak_mem_mb if graph.peak_mem_mb is not None else eager.peak_mem_mb
            for m in (eager, graph):
                if m.error:
                    print(f"  ! {m.family} b{m.batch} graphs={m.graphs}: {m.error}", flush=True)
        check = ""
        if equivalence:
            cfg = dataclasses.replace(_config(spec, batches[0], False), lr=3e-3, warmup_steps=5)
            try:
                rep = graph_equivalence(cfg, "cuda", steps=20)
                check = f"  {'PASS' if rep.passed else 'FAIL'} {rep.max_rel_diff:.1e} / {rep.noise_rel_diff:.1e}"
                if out is not None:
                    with open(out, "a") as f:
                        f.write(json.dumps({"family": spec_label(spec), "equivalence_passed": rep.passed,
                                            "max_rel_diff": rep.max_rel_diff, "noise_rel_diff": rep.noise_rel_diff,
                                            "eager": list(rep.eager_losses), "graph": list(rep.graph_losses)}) + "\n")
            except RuntimeError as exc:
                check = f"  ERROR {type(exc).__name__}: {str(exc)[:80]}"
        row = f"{spec_label(spec):<34}" + "".join(cells) + f"{_fmt(mem):>13}" + check
        rows.append(row)
        print(row, flush=True)


def concurrency(family: str, procs: Sequence[int], batch: int, graphs: bool, seconds: float) -> None:
    for k in procs:
        start_at = time.time() + 25.0  # time for every process to import, build, warm up and capture
        cmd = [sys.executable, "-m", "sst.bench", "one", "--family", family, "--batch", str(batch),
               "--seconds", str(seconds), "--start-at", str(start_at)] + (["--graphs"] if graphs else [])
        ps = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(k)]
        rates: list[float] = []
        for p in ps:
            stdout, _ = p.communicate()
            parsed: JSONValue = json.loads(stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                v = parsed.get("steps_per_s")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rates.append(float(v))
        total = sum(rates)
        print(f"{family:<34} graphs={graphs!s:<5} procs={k}  per-process "
              f"{', '.join(f'{r:.1f}' for r in rates)}  aggregate {total:.1f} steps/s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sst.bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--batches", default="128,256")
    t.add_argument("--seconds", type=float, default=3.0)
    t.add_argument("--out", default=None)
    t.add_argument("--families", default=None, help="comma-separated labels (default: the full bench set)")
    t.add_argument("--no-check", action="store_true", help="skip the graph equivalence check")
    o = sub.add_parser("one")
    o.add_argument("--family", required=True)
    o.add_argument("--batch", type=int, default=128)
    o.add_argument("--seconds", type=float, default=10.0)
    o.add_argument("--graphs", action="store_true")
    o.add_argument("--start-at", type=float, default=None)
    c = sub.add_parser("concurrency")
    c.add_argument("--family", required=True)
    c.add_argument("--procs", default="1,2,3")
    c.add_argument("--batch", type=int, default=128)
    c.add_argument("--seconds", type=float, default=15.0)
    c.add_argument("--graphs", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "table":
        fams = bench_families() if args.families is None else [parse_family(f) for f in args.families.split(";")]
        table([int(b) for b in args.batches.split(",")], args.seconds, args.out, fams, not args.no_check)
    elif args.cmd == "one":
        m = measure(parse_family(args.family), args.batch, args.graphs, args.seconds, start_at=args.start_at)
        print(json.dumps(m.to_dict()))
    else:
        concurrency(args.family, [int(k) for k in args.procs.split(",")], args.batch, args.graphs, args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
