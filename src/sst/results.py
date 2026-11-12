"""Run records: one JSON line per finished run in ``runs.jsonl``.

Writing
-------
:func:`append_run` serialises a :class:`RunResult` to a single line of strict
JSON (NaN and inf become ``null``) and writes it with one ``os.write`` on a
file opened with ``O_APPEND``, holding an exclusive ``flock`` for the write.
``O_APPEND`` makes the offset update atomic; the lock also keeps two writers'
bytes apart on file systems where large appends are not atomic. Concurrent
workers therefore never interleave lines.

Reading
-------
:func:`load_runs` returns every record in file order. A final line without a
trailing newline that fails to parse is a torn write (the writer died or the
disk filled) and is skipped; a bad line anywhere else raises ``ValueError``.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sst.config import RunConfig, config_from_dict, config_to_dict
from sst.evaluation import EvalResult, JSONValue

Status = Literal["ok", "diverged"]
#: Where a diverged run first produced non-finite values.
DivergedStage = Literal["train", "val_curve", "final_val", "final_eval"]

Curve = tuple[tuple[int, float | None], ...]

_STATUSES: dict[str, Status] = {"ok": "ok", "diverged": "diverged"}
_STAGES: dict[str, DivergedStage] = {
    "train": "train",
    "val_curve": "val_curve",
    "final_val": "final_val",
    "final_eval": "final_eval",
}


@dataclass(frozen=True)
class RunResult:
    """Everything recorded about one finished run.

    ``train_curve`` holds (step, mean train loss over the preceding window);
    ``val_curve`` holds (step, in-distribution val token accuracy at length L).
    ``val`` and ``eval`` are the final evaluations at ``eval_multiplier * L`` on
    the val and eval splits; both are ``None`` for a diverged run.
    """

    config: RunConfig
    status: Status
    diverged_stage: DivergedStage | None
    diverged_at_step: int | None
    steps_completed: int
    train_curve: Curve
    val_curve: Curve
    val: EvalResult | None
    eval: EvalResult | None
    param_count: int
    wall_time_s: float
    peak_gpu_mem_bytes: int | None
    device: str
    cuda_graphs_used: bool
    git_commit: str
    git_dirty: bool
    torch_version: str
    finished_at: str

    @property
    def run_id(self) -> str:
        return self.config.run_id

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "run_id": self.run_id,
            "config": config_to_dict(self.config),
            "status": self.status,
            "diverged_stage": self.diverged_stage,
            "diverged_at_step": self.diverged_at_step,
            "steps_completed": self.steps_completed,
            "train_curve": _curve_to_json(self.train_curve),
            "val_curve": _curve_to_json(self.val_curve),
            "val": None if self.val is None else self.val.to_dict(),
            "eval": None if self.eval is None else self.eval.to_dict(),
            "param_count": self.param_count,
            "wall_time_s": _finite_or_none(self.wall_time_s),
            "peak_gpu_mem_bytes": self.peak_gpu_mem_bytes,
            "device": self.device,
            "cuda_graphs_used": self.cuda_graphs_used,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "torch_version": self.torch_version,
            "finished_at": self.finished_at,
        }

    @staticmethod
    def from_dict(d: dict[str, JSONValue]) -> RunResult:
        cfg = d["config"]
        if not isinstance(cfg, dict):
            raise ValueError("config must be an object")
        config = config_from_dict(cfg)
        if d.get("run_id") != config.run_id:
            raise ValueError(f"run_id {d.get('run_id')!r} does not match its config ({config.run_id})")
        status = _STATUSES.get(_str(d["status"]))
        if status is None:
            raise ValueError(f"unknown status {d['status']!r}")
        stage_raw = d["diverged_stage"]
        stage = None if stage_raw is None else _STAGES.get(_str(stage_raw))
        if stage_raw is not None and stage is None:
            raise ValueError(f"unknown diverged_stage {stage_raw!r}")
        wall = d["wall_time_s"]
        return RunResult(
            config=config,
            status=status,
            diverged_stage=stage,
            diverged_at_step=_opt_int(d["diverged_at_step"]),
            steps_completed=_int(d["steps_completed"]),
            train_curve=_curve_from_json(d["train_curve"]),
            val_curve=_curve_from_json(d["val_curve"]),
            val=_eval_from_json(d["val"]),
            eval=_eval_from_json(d["eval"]),
            param_count=_int(d["param_count"]),
            wall_time_s=float(wall) if isinstance(wall, (int, float)) and not isinstance(wall, bool) else math.nan,
            peak_gpu_mem_bytes=_opt_int(d["peak_gpu_mem_bytes"]),
            device=_str(d["device"]),
            cuda_graphs_used=_bool(d["cuda_graphs_used"]),
            git_commit=_str(d["git_commit"]),
            git_dirty=_bool(d["git_dirty"]),
            torch_version=_str(d["torch_version"]),
            finished_at=_str(d["finished_at"]),
        )


# ------------------------------------------------------------------ helpers


def _finite_or_none(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _curve_to_json(curve: Curve) -> list[JSONValue]:
    return [[step, _finite_or_none(v)] for step, v in curve]


def _curve_from_json(v: JSONValue) -> Curve:
    if not isinstance(v, list):
        raise ValueError("curve must be a list")
    out: list[tuple[int, float | None]] = []
    for point in v:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"bad curve point {point!r}")
        step, value = point
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"bad curve value {value!r}")
        out.append((_int(step), None if value is None else float(value)))
    return tuple(out)


def _eval_from_json(v: JSONValue) -> EvalResult | None:
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("eval result must be an object or null")
    return EvalResult.from_dict(v)


def _int(v: JSONValue) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"expected int, got {v!r}")
    return v


def _opt_int(v: JSONValue) -> int | None:
    return None if v is None else _int(v)


def _str(v: JSONValue) -> str:
    if not isinstance(v, str):
        raise ValueError(f"expected string, got {v!r}")
    return v


def _bool(v: JSONValue) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"expected bool, got {v!r}")
    return v


# ------------------------------------------------------------------ file I/O


def default_results_dir() -> Path:
    """``$SST_RESULTS_DIR`` if set, else ``results/`` in the repository's main checkout.

    Using the main checkout (the parent of git's common dir) rather than the
    current worktree means every worktree and session shares one ``runs.jsonl``
    and one queue. Falls back to ``./results`` outside a git repository.
    """
    env = os.environ.get("SST_RESULTS_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(common).parent / "results"
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd() / "results"


def encode_line(result: RunResult) -> bytes:
    text = json.dumps(result.to_dict(), allow_nan=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def append_run(path: Path, result: RunResult) -> None:
    """Append ``result`` as one line, atomically with respect to other appenders."""
    data = encode_line(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            written = os.write(fd, data)
            if written != len(data):
                raise OSError(f"short write to {path}: {written} of {len(data)} bytes")
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def load_runs(path: Path) -> list[RunResult]:
    """All records in file order; see the module docstring for torn lines."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    torn_tail = not raw.endswith(b"\n")
    out: list[RunResult] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            parsed: JSONValue = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("record is not an object")
            out.append(RunResult.from_dict(parsed))
        except (ValueError, KeyError) as exc:
            if torn_tail and i == len(lines) - 1:
                continue
            raise ValueError(f"{path}: bad record on line {i + 1}: {exc}") from exc
    return out


def completed_run_ids(path: Path) -> set[str]:
    """Run ids that have a record, read without full parsing (fast for large files)."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    raw = path.read_bytes()
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            parsed: JSONValue = json.loads(line)
        except ValueError:
            continue  # torn tail; load_runs reports bad middle lines
        if isinstance(parsed, dict):
            rid = parsed.get("run_id")
            if isinstance(rid, str):
                ids.add(rid)
    return ids
