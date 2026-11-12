"""End to end on CPU: plan file -> submit -> worker -> runs.jsonl -> analysis, through the real CLIs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sst.analysis import analyse, format_table
from sst.config import RunConfig, config_to_dict
from sst.models import ModelSpec
from sst.queue import JobQueue
from sst.results import load_runs


def _configs() -> list[RunConfig]:
    common = dict(steps=50, batch_size=16, warmup_steps=10, log_every=10, val_every=25,
                  val_sequences=64, eval_sequences=64, curve_val_sequences=32)
    return [
        RunConfig(task="parity", model=ModelSpec("lstm", d_model=32), lr=3e-3, seed=0, **common),  # type: ignore[arg-type]
        RunConfig(task="flipflop", model=ModelSpec("diag_pos", d_model=32), lr=3e-3, seed=0, **common),  # type: ignore[arg-type]
    ]


def _cli(results: Path, *args: str) -> str:
    out = subprocess.run([sys.executable, "-m", "sst.queue", "--results", str(results), *args],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


@pytest.mark.slow
def test_submit_work_record_analyse(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    configs = _configs()
    plan.write_text(json.dumps([config_to_dict(c) for c in configs]))

    assert "submitted 2" in _cli(tmp_path, "submit", str(plan), "--priority", "80", "--note", "smoke")
    assert "submitted 0" in _cli(tmp_path, "submit", str(plan))  # resubmitting is a no-op
    _cli(tmp_path, "work", "--slots", "2", "--device", "cpu", "--drain", "--poll", "0.2")
    assert JobQueue(tmp_path).counts() == {"pending": 0, "running": 0, "done": 2, "failed": 0}

    runs = load_runs(tmp_path / "runs.jsonl")
    assert sorted(r.run_id for r in runs) == sorted(c.run_id for c in configs)
    for r in runs:
        assert r.status == "ok", r
        assert r.steps_completed == 50
        assert r.val is not None and r.eval is not None and r.eval.eval_length == 512
        assert len(r.train_curve) == 5 and len(r.val_curve) == 2
        assert (tmp_path / "checkpoints" / f"{r.run_id}.pt").exists()

    # A second worker pass finds nothing to do, and a resubmit after completion is skipped.
    assert "skipped 2 already recorded" in _cli(tmp_path, "submit", str(plan))

    cells = analyse(runs, lrs=(3e-3,), n_seeds=1)
    assert sorted((c.family, c.task) for c in cells) == [("diag_pos[d_model=32]", "flipflop"), ("lstm[d_model=32]", "parity")]
    assert all(c.category in ("LG", "LG-some", "ID-only", "F") for c in cells)
    assert "flipflop" in format_table(cells)
