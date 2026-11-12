"""Directory job queue: submit, dedupe, priority, atomic claim, stale recovery, the worker loop."""

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pytest

from sst.config import RunConfig
from sst.models import ModelSpec
from sst.queue import JobQueue, Worker, WorkerLockedError, load_job


def _cfg(seed: int, lr: float = 1e-3) -> RunConfig:
    return RunConfig(task="parity", model=ModelSpec("lstm", d_model=16, n_heads=2), lr=lr, seed=seed, steps=10)


def _record(root: Path, run_id: str) -> None:
    with (root / "runs.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": run_id}) + "\n")


def _names(d: Path) -> list[str]:
    return sorted(p.name for p in d.glob("*.json"))


def test_submit_writes_pending_jobs_that_round_trip(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    report = q.submit([_cfg(0), _cfg(1)], priority=70, note="sweep")
    assert report.submitted == 2
    jobs = [load_job(p) for p in sorted(q.pending.glob("*.json"))]
    assert {j.config for j in jobs} == {_cfg(0), _cfg(1)}
    assert all(j.priority == 70 and j.note == "sweep" and j.attempts == 0 for j in jobs)
    assert not list(q.tmp.iterdir())


def test_submit_skips_recorded_queued_and_running_runs(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    _record(tmp_path, _cfg(0).run_id)
    q.submit([_cfg(1), _cfg(2)])
    claimed = q.claim()
    assert claimed is not None
    report = q.submit([_cfg(0), _cfg(1), _cfg(2), _cfg(3), _cfg(3)])
    assert report.submitted == 1
    assert report.skipped_recorded == 1
    assert report.skipped_queued == 3  # one pending, one running, one duplicate within the plan
    assert len(_names(q.pending)) + len(_names(q.running)) == 3


def test_higher_priority_resubmit_moves_a_pending_job_ahead(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0), _cfg(1)], priority=50)
    report = q.submit([_cfg(1)], priority=90)
    assert report.reprioritised == 1 and report.submitted == 0
    assert len(_names(q.pending)) == 2
    first = q.claim()
    assert first is not None and load_job(first).config == _cfg(1)
    assert load_job(first).priority == 90
    # A lower priority never demotes a job.
    q.submit([_cfg(0)], priority=10)
    second = q.claim()
    assert second is not None and load_job(second).priority == 50


def test_claim_order_is_priority_then_fifo(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)], priority=50)
    q.submit([_cfg(1)], priority=50)
    q.submit([_cfg(2)], priority=80)
    q.submit([_cfg(3)], priority=10)
    order: list[int] = []
    while (p := q.claim()) is not None:
        order.append(load_job(p).config.seed)
    assert order == [2, 0, 1, 3]
    assert len(_names(q.running)) == 4


def test_priority_out_of_range_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JobQueue(tmp_path).submit([_cfg(0)], priority=100)


def _claimer(root: str, out: str) -> None:
    q = JobQueue(Path(root))
    got: list[str] = []
    while (p := q.claim()) is not None:
        got.append(p.name)
    Path(out).write_text("\n".join(got))


def test_concurrent_claimers_take_each_job_exactly_once(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(s) for s in range(60)])
    ctx = mp.get_context("spawn")
    outs = [tmp_path / f"claimed{i}.txt" for i in range(4)]
    procs = [ctx.Process(target=_claimer, args=(str(tmp_path), str(o))) for o in outs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    claimed = [line for o in outs for line in o.read_text().splitlines() if line]
    assert len(claimed) == 60
    assert len(set(claimed)) == 60


def _submitter(root: str, seeds: list[int]) -> None:
    JobQueue(Path(root)).submit([_cfg(s) for s in seeds])


def test_concurrent_submitters_never_duplicate_a_run(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_submitter, args=(str(tmp_path), list(range(30)))) for _ in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    q = JobQueue(tmp_path)
    ids = [load_job(p).config.run_id for p in q.pending.glob("*.json")]
    assert len(ids) == 30 and len(set(ids)) == 30


def test_recover_stale_running_jobs(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0), _cfg(1)])
    a, b = q.claim(), q.claim()
    assert a is not None and b is not None
    b_id = load_job(b).config.run_id
    _record(tmp_path, load_job(a).config.run_id)
    moved = q.recover_stale()
    assert moved == (1, 1)  # (to done, to pending)
    assert _names(q.done) == [a.name]
    assert [load_job(p).config.run_id for p in q.pending.glob("*.json")] == [b_id]
    assert not _names(q.running)


def test_resubmitting_a_failed_job_requeues_it(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)])
    p = q.claim()
    assert p is not None
    p.rename(q.failed / p.name)
    report = q.submit([_cfg(0)])
    assert report.submitted == 1
    assert not _names(q.failed) and len(_names(q.pending)) == 1


def test_only_one_worker_holds_the_lock(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    with q.worker_lock():
        with pytest.raises(WorkerLockedError):
            with JobQueue(tmp_path).worker_lock():
                pass
    with q.worker_lock():
        pass


# ------------------------------------------------------------- worker loop


def _fake_command(root: Path, sleep: float, exit_code: int = 0, write_record: bool = True) -> "FakeCmd":
    return FakeCmd(root, sleep, exit_code, write_record)


class FakeCmd:
    """Stands in for `python -m sst.queue run-job`: sleeps, maybe records, exits."""

    def __init__(self, root: Path, sleep: float, exit_code: int, write_record: bool) -> None:
        self.root, self.sleep, self.exit_code, self.write_record = root, sleep, exit_code, write_record

    def __call__(self, job_path: Path) -> list[str]:
        rid = load_job(job_path).config.run_id
        code = (
            "import json, time, sys\n"
            f"time.sleep({self.sleep})\n"
            f"print('working on {rid}')\n"
        )
        if self.write_record:
            code += f"open({str(self.root / 'runs.jsonl')!r}, 'a').write(json.dumps({{'run_id': {rid!r}}}) + '\\n')\n"
        code += f"sys.exit({self.exit_code})\n"
        return [sys.executable, "-c", code]


def test_worker_runs_jobs_concurrently_and_moves_them_to_done(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(s) for s in range(4)])
    worker = Worker(q, slots=2, command=_fake_command(tmp_path, 0.6), poll_seconds=0.05)
    t0 = time.monotonic()
    worker.run(drain=True)
    elapsed = time.monotonic() - t0
    assert len(_names(q.done)) == 4
    assert not _names(q.pending) and not _names(q.running) and not _names(q.failed)
    assert worker.max_active == 2
    assert elapsed < 4 * 0.6  # ran two at a time
    assert (q.logs / f"{_cfg(0).run_id}.log").read_text().startswith("working on")


def test_worker_moves_crashed_jobs_to_failed(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)])
    Worker(q, slots=1, command=_fake_command(tmp_path, 0.0, exit_code=3, write_record=False), poll_seconds=0.05).run(drain=True)
    assert len(_names(q.failed)) == 1 and not _names(q.done)


def test_worker_treats_clean_exit_without_record_as_failure(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)])
    Worker(q, slots=1, command=_fake_command(tmp_path, 0.0, exit_code=0, write_record=False), poll_seconds=0.05).run(drain=True)
    assert len(_names(q.failed)) == 1


def test_worker_skips_jobs_recorded_after_submission(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)])
    _record(tmp_path, _cfg(0).run_id)
    cmd = _fake_command(tmp_path, 0.0)
    worker = Worker(q, slots=1, command=cmd, poll_seconds=0.05)
    worker.run(drain=True)
    assert worker.launched == 0
    assert len(_names(q.done)) == 1


def test_worker_retries_out_of_memory_once(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)])

    def oom(job_path: Path) -> list[str]:
        return [sys.executable, "-c", "import sys; print('torch.OutOfMemoryError: CUDA out of memory'); sys.exit(1)"]

    worker = Worker(q, slots=1, command=oom, poll_seconds=0.05, max_attempts=2)
    worker.run(drain=True)
    assert worker.launched == 2
    failed = list(q.failed.glob("*.json"))
    assert len(failed) == 1 and load_job(failed[0]).attempts == 2


def test_worker_recovers_stale_jobs_on_start(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([_cfg(0)])
    assert q.claim() is not None  # a previous worker died holding it
    worker = Worker(q, slots=1, command=_fake_command(tmp_path, 0.0), poll_seconds=0.05)
    worker.run(drain=True)
    assert worker.launched == 1 and len(_names(q.done)) == 1
