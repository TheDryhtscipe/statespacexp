"""Training-step pieces: the masked loss and the learning-rate schedule."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from sst.train import lr_at, masked_cross_entropy


def _rand(b: int, t: int, c: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, c, generator=g), torch.randint(0, c, (b, t), generator=g)


def test_full_mask_equals_mean_cross_entropy() -> None:
    logits, targets = _rand(3, 5, 4)
    mask = torch.ones(3, 5, dtype=torch.bool)
    expected = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    assert torch.allclose(masked_cross_entropy(logits, targets, mask), expected)


def test_loss_is_mean_over_masked_positions_only() -> None:
    logits, targets = _rand(2, 6, 3, seed=1)
    mask = torch.tensor([[1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    per = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="none").reshape(2, 6)
    expected = per[mask].mean()
    assert torch.allclose(masked_cross_entropy(logits, targets, mask), expected)


def test_unmasked_positions_have_no_effect_on_loss_or_gradient() -> None:
    logits, targets = _rand(2, 6, 3, seed=2)
    mask = torch.tensor([[1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    a = logits.clone().requires_grad_(True)
    b = logits.clone()
    b[~mask] = float("nan")  # even NaN at unscored positions must not leak in
    b.requires_grad_(True)
    la = masked_cross_entropy(a, targets, mask)
    lb = masked_cross_entropy(b, targets, mask)
    assert torch.allclose(la, lb)
    la.backward()  # type: ignore[no-untyped-call]
    lb.backward()  # type: ignore[no-untyped-call]
    assert a.grad is not None and b.grad is not None
    assert torch.all(a.grad[~mask] == 0)
    assert torch.all(b.grad[~mask] == 0)
    assert torch.allclose(a.grad[mask], b.grad[mask])


def test_empty_mask_gives_zero_loss_not_nan() -> None:
    logits, targets = _rand(1, 4, 2)
    loss = masked_cross_entropy(logits, targets, torch.zeros(1, 4, dtype=torch.bool))
    assert loss.item() == 0.0


def test_lr_warmup_is_linear_and_peaks_at_end_of_warmup() -> None:
    assert lr_at(0, peak=1.0, warmup=10, total=100) == pytest.approx(0.1)
    assert lr_at(4, peak=1.0, warmup=10, total=100) == pytest.approx(0.5)
    assert lr_at(9, peak=1.0, warmup=10, total=100) == pytest.approx(1.0)


def test_lr_cosine_decays_to_zero() -> None:
    lrs = [lr_at(s, peak=2.0, warmup=10, total=110) for s in range(110)]
    assert lrs[10] == pytest.approx(2.0)
    assert lrs[60] == pytest.approx(1.0)
    assert all(x >= y for x, y in zip(lrs[9:], lrs[10:], strict=False))
    assert 0.0 < lrs[-1] < 2.0 * (1 - math.cos(math.pi / 100))
    assert lr_at(110, peak=2.0, warmup=10, total=110) == pytest.approx(0.0)


def test_lr_with_warmup_longer_than_run() -> None:
    assert lr_at(4, peak=1.0, warmup=500, total=50) == pytest.approx(5 / 500)


# ------------------------------------------------------------- train() on CPU

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import sst.train as train_mod  # noqa: E402
from sst.config import RunConfig  # noqa: E402
from sst.evaluation import EvalResult, NonFiniteLogitsError, make_rng  # noqa: E402
from sst.evaluation import evaluate as real_evaluate  # noqa: E402
from sst.models import ModelSpec  # noqa: E402
from sst.results import load_runs  # noqa: E402
from sst.tasks import get_task  # noqa: E402
from sst.train import TrainStream, run_and_record, train  # noqa: E402


def _tiny(task: str = "parity", seed: int = 0, steps: int = 6) -> RunConfig:
    return RunConfig(
        task=task,
        model=ModelSpec("lstm", d_model=16, n_heads=2, n_layers=1),
        lr=1e-2,
        seed=seed,
        steps=steps,
        batch_size=4,
        train_length=8,
        warmup_steps=2,
        val_sequences=16,
        eval_sequences=24,
        log_every=2,
        val_every=3,
        curve_val_sequences=8,
        data_chunk_steps=4,
        eval_batch_size=8,
    )


def test_train_stream_draws_from_the_train_split_in_chunks() -> None:
    cfg = _tiny()
    task = get_task(cfg.task)
    stream = iter(TrainStream(task, cfg, torch.device("cpu")))
    batches = [next(stream) for _ in range(6)]
    rng = make_rng(cfg.task, "train", cfg.seed)
    chunk1 = task.sample(16, 8, rng)
    chunk2 = task.sample(16, 8, rng)
    np.testing.assert_array_equal(batches[0][0].numpy(), chunk1.inputs[0:4])
    np.testing.assert_array_equal(batches[3][1].numpy(), chunk1.targets[12:16])
    np.testing.assert_array_equal(batches[4][0].numpy(), chunk2.inputs[0:4])
    np.testing.assert_array_equal(batches[5][2].numpy(), chunk2.mask[4:8])


def test_train_records_curves_and_both_splits(tmp_path: Path) -> None:
    cfg = _tiny()
    r = train(cfg, "cpu", tmp_path)
    assert r.status == "ok" and r.diverged_stage is None
    assert r.steps_completed == 6
    assert [s for s, _ in r.train_curve] == [2, 4, 6]
    assert [s for s, _ in r.val_curve] == [3, 6]
    assert r.val is not None and r.eval is not None
    assert (r.val.split, r.eval.split) == ("val", "eval")
    assert (r.val.num_sequences, r.eval.num_sequences) == (16, 24)
    assert r.val.eval_length == r.eval.eval_length == 64
    assert (r.val.seed, r.eval.seed) == (0, 0)
    assert r.param_count > 0 and r.wall_time_s > 0 and r.peak_gpu_mem_bytes is None
    ckpt = train_mod.checkpoint_path(tmp_path, cfg.run_id)
    state = torch.load(ckpt)
    assert "readout.weight" in state


def test_eval_sets_do_not_depend_on_training_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int]] = []
    real = real_evaluate

    def spy(*args: object, **kwargs: object) -> EvalResult:
        split = kwargs["split"]
        seed = kwargs["seed"]
        assert isinstance(split, str) and isinstance(seed, int)
        seen.append((split, seed))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_mod, "evaluate", spy)
    train(_tiny(seed=3), "cpu")
    train(_tiny(seed=4), "cpu")
    assert seen and all(seed == 0 for _, seed in seen)
    assert {split for split, _ in seen} == {"val", "eval"}


def test_nan_loss_records_diverged_instead_of_crashing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = train_mod.masked_cross_entropy
    calls = [0]

    def nan_after_three(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        loss = real(logits, targets, mask)
        return loss * float("nan") if calls[0] >= 3 else loss

    monkeypatch.setattr(train_mod, "masked_cross_entropy", nan_after_three)
    r = train(_tiny(steps=10), "cpu", tmp_path)
    assert r.status == "diverged"
    assert r.diverged_stage == "train"
    # NaN at step 3; step 3 is a val-curve step, which checks the loss flag first.
    assert r.diverged_at_step == 3
    assert r.steps_completed == 3
    assert r.val is None and r.eval is None
    assert r.train_curve[-1] == (3, None)
    assert r.val_curve == ()
    assert not train_mod.checkpoint_path(tmp_path, r.run_id).exists()


def test_non_finite_eval_logits_record_diverged(monkeypatch: pytest.MonkeyPatch) -> None:
    real = real_evaluate

    def boom(*args: object, **kwargs: object) -> EvalResult:
        if kwargs["split"] == "eval":
            raise NonFiniteLogitsError("logits contain NaN or inf")
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_mod, "evaluate", boom)
    r = train(_tiny(), "cpu")
    assert r.status == "diverged" and r.diverged_stage == "final_eval"
    assert r.val is None and r.eval is None


def test_flipflop_trains_on_reads_only(monkeypatch: pytest.MonkeyPatch) -> None:
    masks: list[torch.Tensor] = []
    real = train_mod.masked_cross_entropy

    def spy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masks.append(mask.clone())
        return real(logits, targets, mask)

    monkeypatch.setattr(train_mod, "masked_cross_entropy", spy)
    train(_tiny(task="flipflop", steps=2), "cpu")
    assert masks and not bool(masks[0].all()) and bool(masks[0].any())


def test_run_and_record_appends(tmp_path: Path) -> None:
    r = run_and_record(_tiny(), "cpu", tmp_path)
    assert [x.run_id for x in load_runs(tmp_path / "runs.jsonl")] == [r.run_id]


def test_same_config_same_result_on_cpu() -> None:
    a, b = train(_tiny(), "cpu"), train(_tiny(), "cpu")
    assert a.train_curve == b.train_curve
    assert a.eval == b.eval


def test_nan_loss_is_caught_at_the_next_log_step(monkeypatch: pytest.MonkeyPatch) -> None:
    real = train_mod.masked_cross_entropy
    calls = [0]

    def nan_from_three(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        loss = real(logits, targets, mask)
        return loss * float("nan") if calls[0] >= 3 else loss

    monkeypatch.setattr(train_mod, "masked_cross_entropy", nan_from_three)
    cfg = dataclasses.replace(_tiny(steps=10), val_every=100)
    r = train(cfg, "cpu")
    assert (r.status, r.diverged_stage, r.diverged_at_step) == ("diverged", "train", 4)
    assert r.train_curve == ((2, r.train_curve[0][1]), (4, None))
    assert r.train_curve[0][1] is not None
