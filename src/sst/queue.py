"""A directory job queue for one shared GPU.

Layout under ``<results>/queue/``::

    pending/   jobs waiting to run
    running/   jobs a worker has claimed
    done/      jobs whose run has a record in <results>/runs.jsonl
    failed/    jobs whose process crashed or exited without a record
    tmp/       staging for atomic writes
    logs/      <run_id>.log, stdout and stderr of each job's process

A job is one JSON file holding a serialised :class:`~sst.config.RunConfig`, a
priority (0-99, higher runs first), a free-text note and an attempt count. Its
file name is ``{99 - priority:02d}-{submit time in ns:020d}-{run_id}.json``, so
sorting names gives the run order: priority first, then first-in first-out.

Concurrency
-----------
* A worker claims a job with one atomic ``os.rename`` from ``pending/`` to ``running/``.
* Every state change (submit, claim, requeue, finish) also holds an exclusive
  ``flock`` on ``queue/queue.lock``. That makes a submitter's duplicate check
  (recorded? pending? running?) see a consistent picture, so any number of
  sessions can submit the same plan and each run is queued once.
* Only one worker may run: it holds ``flock`` on ``queue/worker.lock`` for its
  lifetime (released by the kernel if it dies). That is what makes it safe for
  a starting worker to treat every job in ``running/`` as stale: jobs whose run
  has a record move to ``done/``, the rest back to ``pending/``.
* Each job runs in its own process (``python -m sst.queue run-job``), so a crash
  or out-of-memory error kills only that job. The child writes the record to
  ``runs.jsonl`` itself before exiting; the worker moves the job to ``done/`` only
  if exit code is 0 *and* the record exists. Children get SIGTERM if the worker
  dies (``PR_SET_PDEATHSIG``), so a restarted worker never races an orphan.
* Before launching, the worker re-checks ``runs.jsonl`` and its own running set,
  so a job recorded after it was submitted is never run twice.
* A job that fails with CUDA out-of-memory (the GPU is shared) is put back in
  ``pending/`` until it has had ``max_attempts`` attempts.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from sst.config import RunConfig, config_from_dict, config_to_dict
from sst.evaluation import JSONValue
from sst.results import completed_run_ids, default_results_dir

DEFAULT_PRIORITY = 50
PILOT_PRIORITY = 80
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED")


class WorkerLockedError(RuntimeError):
    """Another worker already holds the queue."""


@dataclass(frozen=True)
class Job:
    config: RunConfig
    priority: int = DEFAULT_PRIORITY
    note: str = ""
    attempts: int = 0
    submitted_at_ns: int = 0

    @property
    def run_id(self) -> str:
        return self.config.run_id

    def file_name(self) -> str:
        return f"{99 - self.priority:02d}-{self.submitted_at_ns:020d}-{self.run_id}.json"

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "run_id": self.run_id,
            "config": config_to_dict(self.config),
            "priority": self.priority,
            "note": self.note,
            "attempts": self.attempts,
            "submitted_at_ns": self.submitted_at_ns,
        }


def _int_field(d: dict[str, JSONValue], key: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"job field {key!r} must be an int, got {v!r}")
    return v


def job_from_dict(d: dict[str, JSONValue]) -> Job:
    cfg = d.get("config")
    note = d.get("note", "")
    if not isinstance(cfg, dict) or not isinstance(note, str):
        raise ValueError("job needs an object 'config' and a string 'note'")
    return Job(
        config=config_from_dict(cfg),
        priority=_int_field(d, "priority"),
        note=note,
        attempts=_int_field(d, "attempts"),
        submitted_at_ns=_int_field(d, "submitted_at_ns"),
    )


def load_job(path: Path) -> Job:
    parsed: JSONValue = json.loads(path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: job is not a JSON object")
    return job_from_dict(parsed)


def run_id_of(path: Path) -> str:
    """The run id from a job file name (no parsing of the file)."""
    return path.stem.rsplit("-", 1)[-1]


@dataclass
class SubmitReport:
    submitted: int = 0
    skipped_recorded: int = 0
    skipped_queued: int = 0
    reprioritised: int = 0
    requeued_failed: int = 0

    def __str__(self) -> str:
        return (
            f"submitted {self.submitted} (of which {self.requeued_failed} previously failed), "
            f"skipped {self.skipped_recorded} already recorded and {self.skipped_queued} already queued, "
            f"reprioritised {self.reprioritised}"
        )


class JobQueue:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_path = root / "runs.jsonl"
        q = root / "queue"
        self.dir = q
        self.pending = q / "pending"
        self.running = q / "running"
        self.done = q / "done"
        self.failed = q / "failed"
        self.tmp = q / "tmp"
        self.logs = q / "logs"
        for d in (self.pending, self.running, self.done, self.failed, self.tmp, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- locks

    @contextlib.contextmanager
    def _flock(self, name: str, blocking: bool = True) -> Iterator[None]:
        fd = os.open(self.dir / name, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise WorkerLockedError(f"another worker holds {self.dir / name}") from exc
            yield
        finally:
            os.close(fd)  # closing releases the lock

    def state_lock(self) -> contextlib.AbstractContextManager[None]:
        return self._flock("queue.lock")

    def worker_lock(self) -> contextlib.AbstractContextManager[None]:
        """Held by the single worker for its lifetime; raises :class:`WorkerLockedError` if taken."""
        return self._flock("worker.lock", blocking=False)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _jobs(d: Path) -> list[Path]:
        return sorted(d.glob("*.json"))

    def _write(self, job: Job, dest_dir: Path) -> Path:
        tmp = self.tmp / f"{job.file_name()}.{os.getpid()}"
        tmp.write_text(json.dumps(job.to_dict(), allow_nan=False))
        dest = dest_dir / job.file_name()
        os.rename(tmp, dest)
        return dest

    # ---------------------------------------------------------------- submit

    def submit(self, configs: Sequence[RunConfig], priority: int = DEFAULT_PRIORITY, note: str = "") -> SubmitReport:
        """Queue each config unless its run is recorded or already queued.

        A queued job with a lower priority is moved ahead; a higher one is left alone.
        A job in ``failed/`` is moved back to ``pending/`` with zero attempts.
        """
        if not 0 <= priority <= 99:
            raise ValueError(f"priority must be in [0, 99], got {priority}")
        report = SubmitReport()
        with self.state_lock():
            recorded = completed_run_ids(self.runs_path)
            pending = {run_id_of(p): p for p in self._jobs(self.pending)}
            running = {run_id_of(p) for p in self._jobs(self.running)}
            failed = {run_id_of(p): p for p in self._jobs(self.failed)}
            seen: set[str] = set()
            for cfg in configs:
                rid = cfg.run_id
                if rid in recorded:
                    report.skipped_recorded += 1
                    continue
                if rid in seen or rid in running:
                    report.skipped_queued += 1
                    continue
                seen.add(rid)
                if rid in pending:
                    old_path = pending[rid]
                    old = load_job(old_path)
                    if priority > old.priority:
                        new = Job(old.config, priority, old.note, old.attempts, old.submitted_at_ns)
                        self._write(new, self.pending)
                        old_path.unlink()
                        report.reprioritised += 1
                    else:
                        report.skipped_queued += 1
                    continue
                if rid in failed:
                    failed[rid].unlink()
                    report.requeued_failed += 1
                self._write(Job(cfg, priority, note, 0, time.time_ns()), self.pending)
                report.submitted += 1
        return report

    # ---------------------------------------------------------------- claim and moves

    def claim(self) -> Path | None:
        """Move the first pending job to ``running/`` with an atomic rename; ``None`` if none."""
        with self.state_lock():
            for path in self._jobs(self.pending):
                dest = self.running / path.name
                try:
                    os.rename(path, dest)
                except FileNotFoundError:
                    continue
                os.utime(dest)  # mtime = claim time, for `status`
                return dest
        return None

    def move(self, path: Path, dest_dir: Path) -> Path:
        with self.state_lock():
            dest = dest_dir / path.name
            os.rename(path, dest)
            return dest

    def requeue(self, path: Path, attempts: int) -> Path:
        """Put a running job back in ``pending/`` (keeping its place) with a new attempt count."""
        job = load_job(path)
        with self.state_lock():
            dest = self._write(Job(job.config, job.priority, job.note, attempts, job.submitted_at_ns), self.pending)
            path.unlink()
            return dest

    def fail(self, path: Path, attempts: int) -> Path:
        job = load_job(path)
        with self.state_lock():
            dest = self._write(Job(job.config, job.priority, job.note, attempts, job.submitted_at_ns), self.failed)
            path.unlink()
            return dest

    def recover_stale(self) -> tuple[int, int]:
        """Move every job in ``running/`` to ``done/`` if recorded, else to ``pending/``.

        Only safe while holding :meth:`worker_lock`. Returns (to done, to pending).
        """
        to_done = to_pending = 0
        with self.state_lock():
            recorded = completed_run_ids(self.runs_path)
            for path in self._jobs(self.running):
                if run_id_of(path) in recorded:
                    os.rename(path, self.done / path.name)
                    to_done += 1
                else:
                    os.rename(path, self.pending / path.name)
                    to_pending += 1
        return to_done, to_pending

    def requeue_failed(self) -> int:
        n = 0
        for path in self._jobs(self.failed):
            with self.state_lock():
                job = load_job(path)
                self._write(Job(job.config, job.priority, job.note, 0, job.submitted_at_ns), self.pending)
                path.unlink()
            n += 1
        return n

    def counts(self) -> dict[str, int]:
        return {d.name: len(self._jobs(d)) for d in (self.pending, self.running, self.done, self.failed)}


# -------------------------------------------------------------------- worker


def _die_with_parent() -> None:
    """In the child, before exec: get SIGTERM when the worker process dies (Linux)."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
    except OSError:
        pass


def default_command(device: str, results_dir: Path) -> Callable[[Path], list[str]]:
    def command(job_path: Path) -> list[str]:
        return [sys.executable, "-m", "sst.queue", "--results", str(results_dir),
                "run-job", str(job_path), "--device", device]

    return command


@dataclass
class _Active:
    proc: subprocess.Popen[bytes]
    path: Path
    run_id: str
    log: IO[bytes]
    started: float


@dataclass
class Worker:
    """Runs up to ``slots`` jobs at once, each in its own process; see the module docstring."""

    queue: JobQueue
    slots: int
    command: Callable[[Path], list[str]]
    poll_seconds: float = 10.0
    max_attempts: int = 2
    verbose: bool = False
    active: list[_Active] = field(default_factory=list)
    max_active: int = 0
    launched: int = 0

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    def _log_tail(self, run_id: str, n_bytes: int = 20000) -> str:
        path = self.queue.logs / f"{run_id}.log"
        if not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - n_bytes))
            return f.read().decode("utf-8", errors="replace")

    def _reap(self) -> None:
        still: list[_Active] = []
        recorded: set[str] | None = None
        for a in self.active:
            rc = a.proc.poll()
            if rc is None:
                still.append(a)
                continue
            a.log.close()
            if recorded is None:
                recorded = completed_run_ids(self.queue.runs_path)
            elapsed = time.monotonic() - a.started
            if rc == 0 and a.run_id in recorded:
                self.queue.move(a.path, self.queue.done)
                self._say(f"done   {a.run_id} in {elapsed:.0f}s")
                continue
            attempts = load_job(a.path).attempts + 1
            oom = any(marker in self._log_tail(a.run_id) for marker in OOM_MARKERS)
            if oom and attempts < self.max_attempts:
                self.queue.requeue(a.path, attempts)
                self._say(f"oom    {a.run_id} (attempt {attempts}); requeued")
            else:
                self.queue.fail(a.path, attempts)
                why = "exit 0 but no record" if rc == 0 else f"exit {rc}"
                self._say(f"failed {a.run_id}: {why}; log {self.queue.logs / (a.run_id + '.log')}")
        self.active = still

    def _fill(self) -> None:
        while len(self.active) < self.slots:
            path = self.queue.claim()
            if path is None:
                return
            rid = run_id_of(path)
            if rid in completed_run_ids(self.queue.runs_path) or rid in {a.run_id for a in self.active}:
                self.queue.move(path, self.queue.done)
                self._say(f"skip   {rid}: already recorded or running")
                continue
            log = (self.queue.logs / f"{rid}.log").open("ab")
            proc = subprocess.Popen(self.command(path), stdout=log, stderr=subprocess.STDOUT,
                                    preexec_fn=_die_with_parent)
            self.active.append(_Active(proc, path, rid, log, time.monotonic()))
            self.launched += 1
            self.max_active = max(self.max_active, len(self.active))
            self._say(f"start  {rid} ({load_job(path).note}); {len(self.active)}/{self.slots} slots busy")

    def _shutdown(self) -> None:
        for a in self.active:
            if a.proc.poll() is None:
                a.proc.terminate()
        for a in self.active:
            try:
                a.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                a.proc.kill()
                a.proc.wait()
            a.log.close()
        self.active = []
        # Anything still in running/ without a record goes back to pending.
        self.queue.recover_stale()

    def run(self, drain: bool = False) -> None:
        """Work until interrupted, or until the queue is empty if ``drain``."""
        with self.queue.worker_lock():
            done, back = self.queue.recover_stale()
            if done or back:
                self._say(f"recovered stale jobs: {done} to done, {back} to pending")
            try:
                while True:
                    self._reap()
                    self._fill()
                    if drain and not self.active and not any(self.queue.pending.glob("*.json")):
                        return
                    time.sleep(self.poll_seconds)
            finally:
                self._shutdown()


# -------------------------------------------------------------------- commands


def run_job(job_path: Path, device: str, results_dir: Path) -> None:
    from sst.train import run_and_record  # torch is imported only in job processes

    job = load_job(job_path)
    print(f"run {job.run_id} ({job.note}) on {device}: {json.dumps(config_to_dict(job.config))}", flush=True)
    result = run_and_record(job.config, device, results_dir)
    print(f"status {result.status} after {result.steps_completed} steps in {result.wall_time_s:.1f}s", flush=True)


def load_plan(path: Path) -> list[RunConfig]:
    """A plan file is a JSON list of serialised RunConfigs (see :func:`sst.config.config_to_dict`)."""
    parsed: JSONValue = json.loads(path.read_text())
    if not isinstance(parsed, list):
        raise ValueError(f"{path}: a plan is a JSON list of configs")
    out: list[RunConfig] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: every plan entry must be an object")
        out.append(config_from_dict(item))
    return out


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sst.queue", description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--results", type=Path, default=None,
                        help="results directory (default: $SST_RESULTS_DIR or <main checkout>/results)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("work", help="run queued jobs")
    w.add_argument("--slots", type=int, default=1)
    w.add_argument("--device", default="cuda")
    w.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    w.add_argument("--poll", type=float, default=10.0)
    w.add_argument("--max-attempts", type=int, default=2)
    s = sub.add_parser("submit", help="queue the configs in a plan file")
    s.add_argument("plan", type=Path)
    s.add_argument("--priority", type=int, default=DEFAULT_PRIORITY)
    s.add_argument("--note", default="")
    r = sub.add_parser("run-job", help="(internal) run one claimed job")
    r.add_argument("job", type=Path)
    r.add_argument("--device", default="cuda")
    sub.add_parser("status", help="count jobs in each state")
    sub.add_parser("requeue-failed", help="move every failed job back to pending")
    args = parser.parse_args(argv)
    results_dir: Path = args.results if args.results is not None else default_results_dir()
    queue = JobQueue(results_dir)

    if args.cmd == "work":
        signal.signal(signal.SIGTERM, _raise_interrupt)
        worker = Worker(queue, args.slots, default_command(args.device, results_dir),
                        poll_seconds=args.poll, max_attempts=args.max_attempts, verbose=True)
        print(f"worker on {results_dir} with {args.slots} slots on {args.device}", flush=True)
        try:
            worker.run(drain=args.drain)
        except KeyboardInterrupt:
            print("interrupted; running jobs returned to pending", flush=True)
            return 130
        except WorkerLockedError as exc:
            print(exc, file=sys.stderr)
            return 1
    elif args.cmd == "submit":
        print(queue.submit(load_plan(args.plan), priority=args.priority, note=args.note))
    elif args.cmd == "run-job":
        run_job(args.job, args.device, results_dir)
    elif args.cmd == "status":
        print(results_dir, json.dumps(queue.counts()))
        for path in JobQueue._jobs(queue.running):
            job = load_job(path)
            age = time.time() - path.stat().st_mtime
            print(f"  running {job.run_id} {job.config.task} {job.config.model.mixer} "
                  f"lr={job.config.lr:g} seed={job.config.seed} ({job.note}) for {age:.0f}s")
    elif args.cmd == "requeue-failed":
        print(f"requeued {queue.requeue_failed()} failed jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
