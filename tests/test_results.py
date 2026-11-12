"""Run records: JSON round trip, atomic appends, and the loader."""

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from sst.config import RunConfig
from sst.evaluation import BucketResult, EvalResult
from sst.models import ModelSpec
from sst.results import RunResult, append_run, completed_run_ids, default_results_dir, load_runs


def _bucket(start: int, end: int, acc: float | None) -> BucketResult:
    return BucketResult(start, end, acc, 10, acc, 2, end, acc, 2)


def _eval(split: str, accs: list[float | None]) -> EvalResult:
    edges = [0, 4, 8, 16, 32]
    return EvalResult(
        task="parity",
        train_length=4,
        eval_length=32,
        num_sequences=2,
        seed=0,
        buckets=tuple(_bucket(edges[i], edges[i + 1], a) for i, a in enumerate(accs)),
        split="val" if split == "val" else "eval",
    )


def make_result(seed: int = 0, status: str = "ok", lr: float = 1e-3) -> RunResult:
    cfg = RunConfig(task="parity", model=ModelSpec("lstm", d_model=32), lr=lr, seed=seed, steps=50)
    ok = status == "ok"
    return RunResult(
        config=cfg,
        status="ok" if ok else "diverged",
        diverged_stage=None if ok else "train",
        diverged_at_step=None if ok else 20,
        steps_completed=50 if ok else 20,
        train_curve=((10, 0.69), (20, None if not ok else 0.5)),
        val_curve=((50, 0.75),),
        val=_eval("val", [1.0, 0.5, None, 0.25]) if ok else None,
        eval=_eval("eval", [0.9, 0.8, 0.7, 0.6]) if ok else None,
        param_count=1234,
        wall_time_s=1.5,
        peak_gpu_mem_bytes=None,
        device="cpu",
        cuda_graphs_used=False,
        git_commit="abc",
        git_dirty=False,
        torch_version="2.x",
        finished_at="2026-11-12T00:00:00Z",
    )


def test_record_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    a, b = make_result(0), make_result(1, status="diverged")
    append_run(path, a)
    append_run(path, b)
    loaded = load_runs(path)
    assert loaded == [a, b]
    assert loaded[0].run_id == a.config.run_id
    assert completed_run_ids(path) == {a.run_id, b.run_id}


def test_each_record_is_one_strict_json_line(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    append_run(path, make_result(1, status="diverged"))
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0], parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert d["run_id"] == make_result(1).run_id
    assert d["status"] == "diverged"


def test_non_finite_curve_values_become_null(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    r = make_result()
    nan_r = RunResult(**{**r.__dict__, "train_curve": ((10, float("nan")), (20, float("inf")))})
    append_run(path, nan_r)
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
    assert load_runs(path)[0].train_curve == ((10, None), (20, None))


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "absent.jsonl") == []
    assert completed_run_ids(tmp_path / "absent.jsonl") == set()


def test_torn_final_line_is_ignored_but_torn_middle_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    append_run(path, make_result(0))
    with path.open("a") as f:
        f.write('{"run_id": "trunc')
    assert len(load_runs(path)) == 1
    with path.open("a") as f:
        f.write("\n")
    append_run(path, make_result(2))
    with pytest.raises(ValueError):
        load_runs(path)


def _writer(path_str: str, seed: int, n: int) -> None:
    for _ in range(n):
        append_run(Path(path_str), make_result(seed))


def test_concurrent_appends_never_interleave(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(str(path), s, 40)) for s in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    lines = path.read_text().splitlines()
    assert len(lines) == 160
    for line in lines:
        json.loads(line)
    assert len(load_runs(path)) == 160


def test_results_dir_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SST_RESULTS_DIR", str(tmp_path))
    assert default_results_dir() == tmp_path


def test_results_dir_is_shared_by_all_worktrees(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    import sst

    monkeypatch.delenv("SST_RESULTS_DIR", raising=False)
    pkg = Path(sst.__file__).resolve().parent
    common = subprocess.run(["git", "-C", str(pkg), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    main_checkout = Path(common).parent
    assert default_results_dir() == main_checkout / "results"
    assert ".claude/worktrees" not in str(default_results_dir())
