"""Hand-built run records for the plan and analysis tests."""

from sst.config import RunConfig
from sst.evaluation import BucketResult, EvalResult, Split
from sst.models import ModelSpec
from sst.results import RunResult

EDGES = (0, 64, 128, 256, 512)


def eval_result(split: Split, accs: tuple[float | None, ...]) -> EvalResult:
    return EvalResult(
        task="s5",
        train_length=64,
        eval_length=512,
        num_sequences=1024,
        seed=0,
        buckets=tuple(
            BucketResult(EDGES[i], EDGES[i + 1], a, 100, a, 10, EDGES[i + 1], a, 10) for i, a in enumerate(accs)
        ),
        split=split,
    )


def config(lr: float = 1e-3, seed: int = 0, mixer: str = "delta", depth: int = 2, task: str = "s5",
           steps: int = 1000) -> RunConfig:
    spec = ModelSpec("delta", n_layers=depth) if mixer == "delta" else ModelSpec("lstm", n_layers=depth)
    return RunConfig(task=task, model=spec, lr=lr, seed=seed, steps=steps)


def record(
    *,
    lr: float = 1e-3,
    seed: int = 0,
    val: tuple[float | None, ...] | None = (1.0, 1.0, 1.0, 1.0),
    ev: tuple[float | None, ...] | None = (1.0, 1.0, 1.0, 1.0),
    diverged: bool = False,
    mixer: str = "delta",
    depth: int = 2,
    task: str = "s5",
    steps: int = 1000,
) -> RunResult:
    return RunResult(
        config=config(lr, seed, mixer, depth, task, steps),
        status="diverged" if diverged else "ok",
        diverged_stage="train" if diverged else None,
        diverged_at_step=100 if diverged else None,
        steps_completed=100 if diverged else steps,
        train_curve=(),
        val_curve=(),
        val=None if diverged or val is None else eval_result("val", val),
        eval=None if diverged or ev is None else eval_result("eval", ev),
        param_count=1,
        wall_time_s=1.0,
        peak_gpu_mem_bytes=None,
        device="cpu",
        cuda_graphs_used=False,
        git_commit="x",
        git_dirty=False,
        torch_version="x",
        finished_at="x",
    )
