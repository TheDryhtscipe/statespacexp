"""RunConfig: defaults, normalisation, a stable run_id, and a JSON round trip."""

import dataclasses
import json
from typing import Literal

import pytest

from sst.config import RunConfig, config_from_dict, config_to_dict, spec_label
from sst.models import DeltaProductOptions, ModelSpec, TTTOptions


def _cfg(
    *,
    task: str = "parity",
    model: ModelSpec | None = None,
    lr: float = 1e-3,
    seed: int = 0,
    steps: int = 1000,
    batch_size: int = 128,
    warmup_steps: int = 500,
    val_sequences: int = -1,
    eval_sequences: int = -1,
    eval_seed: int = 0,
    cuda_graphs: bool = False,
    eval_batch_size: int = 128,
) -> RunConfig:
    return RunConfig(
        task=task,
        model=ModelSpec("lstm") if model is None else model,
        lr=lr,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        val_sequences=val_sequences,
        eval_sequences=eval_sequences,
        eval_seed=eval_seed,
        cuda_graphs=cuda_graphs,
        eval_batch_size=eval_batch_size,
    )


def test_default_eval_sequences_depend_on_task() -> None:
    assert _cfg(task="s5").n_val_sequences == 1024
    assert _cfg(task="s5").n_eval_sequences == 1024
    assert _cfg(task="a5").n_eval_sequences == 1024
    assert _cfg(task="flipflop").n_val_sequences == 4096
    assert _cfg(task="flipflop_sparse").n_eval_sequences == 4096
    assert _cfg(task="flipflop", eval_sequences=100).n_eval_sequences == 100


def test_protocol_defaults() -> None:
    c = _cfg()
    assert (c.batch_size, c.train_length, c.warmup_steps) == (128, 64, 500)
    assert (c.weight_decay, c.grad_clip, c.eval_seed, c.eval_multiplier) == (0.01, 1.0, 0, 8)
    assert c.cuda_graphs is False


def test_unknown_task_rejected() -> None:
    with pytest.raises(ValueError):
        _cfg(task="nope")


def test_run_id_is_stable_and_hex() -> None:
    a = _cfg()
    assert a.run_id == _cfg().run_id
    assert len(a.run_id) == 16
    int(a.run_id, 16)
    # Pinned so that an accidental change to the hashing breaks a test, not the resume logic.
    assert a.run_id == PINNED_RUN_ID


@pytest.mark.parametrize(
    "changed",
    [
        _cfg(lr=3e-3),
        _cfg(seed=1),
        _cfg(steps=999),
        _cfg(task="z3"),
        _cfg(batch_size=256),
        _cfg(model=ModelSpec("lstm", n_layers=3)),
        _cfg(eval_seed=1),
        _cfg(warmup_steps=10),
        _cfg(eval_sequences=10),
        dataclasses.replace(_cfg(), protocol_version=2),
    ],
)
def test_run_id_changes_with_scientific_fields(changed: RunConfig) -> None:
    assert changed.run_id != _cfg().run_id


def test_run_id_ignores_execution_only_fields() -> None:
    assert _cfg(eval_batch_size=32).run_id == _cfg().run_id


def test_cuda_graphs_enters_run_id() -> None:
    """Graph capture changes float rounding, which training amplifies, so it is a scientific field."""
    assert _cfg(cuda_graphs=True).run_id != _cfg().run_id


def test_default_options_and_explicit_defaults_are_the_same_run() -> None:
    implicit = _cfg(model=ModelSpec("deltaproduct"))
    explicit = _cfg(model=ModelSpec("deltaproduct", options=DeltaProductOptions()))
    assert implicit == explicit
    assert implicit.run_id == explicit.run_id
    assert _cfg(model=ModelSpec("deltaproduct", options=DeltaProductOptions(n_householder=4))).run_id != implicit.run_id


def test_replace_task_picks_up_new_default_count() -> None:
    assert dataclasses.replace(_cfg(task="s5"), task="flipflop").n_eval_sequences == 4096


def test_default_sequence_counts_and_explicit_counts_are_the_same_run() -> None:
    assert _cfg(task="flipflop").run_id == _cfg(task="flipflop", val_sequences=4096, eval_sequences=4096).run_id


@pytest.mark.parametrize(
    "spec",
    [
        ModelSpec("lstm"),
        ModelSpec("ttt", options=TTTOptions(inner_model="mlp", minibatch=16, eta_max=0.5)),
        ModelSpec("deltaproduct", d_model=64, n_layers=4, options=DeltaProductOptions(n_householder=4, allow_negative=False)),
    ],
)
def test_json_round_trip(spec: ModelSpec) -> None:
    c = _cfg(model=spec, lr=3e-4, cuda_graphs=True)
    back = config_from_dict(json.loads(json.dumps(config_to_dict(c))))
    assert back == c
    assert back.run_id == c.run_id


def test_from_dict_rejects_wrong_types() -> None:
    d = config_to_dict(_cfg())
    d["seed"] = "zero"
    with pytest.raises(ValueError):
        config_from_dict(d)


def test_spec_label_names_non_default_options() -> None:
    assert spec_label(ModelSpec("lstm")) == "lstm"
    assert spec_label(ModelSpec("lstm", n_layers=4)) == "lstm"
    assert spec_label(ModelSpec("deltaproduct", options=DeltaProductOptions(n_householder=4))) == "deltaproduct[n_householder=4]"
    assert spec_label(ModelSpec("ttt", options=TTTOptions(minibatch=16))) == "ttt[minibatch=16]"
    assert spec_label(ModelSpec("delta", d_model=64)) == "delta[d_model=64]"


PINNED_RUN_ID = "d7bc6724be855e7d"


def test_ttt_default_eta_max_hashes_like_its_resolved_rate() -> None:
    """eta_max=None means the base rate (1.0 linear, 0.1 mlp), so both spellings are one run."""
    cases: tuple[tuple[Literal["linear", "mlp"], float], ...] = (("linear", 1.0), ("mlp", 0.1))
    for inner, rate in cases:
        implicit = RunConfig(task="parity", model=ModelSpec("ttt", options=TTTOptions(inner_model=inner)), lr=1e-3, seed=0, steps=10)
        explicit = RunConfig(task="parity", model=ModelSpec("ttt", options=TTTOptions(inner_model=inner, eta_max=rate)), lr=1e-3, seed=0, steps=10)
        other = RunConfig(task="parity", model=ModelSpec("ttt", options=TTTOptions(inner_model=inner, eta_max=rate / 2)), lr=1e-3, seed=0, steps=10)
        assert implicit.run_id == explicit.run_id
        assert implicit.run_id != other.run_id


def test_ttt_default_eta_max_round_trips_as_none() -> None:
    c = RunConfig(task="parity", model=ModelSpec("ttt", options=TTTOptions(inner_model="mlp")), lr=1e-3, seed=0, steps=10)
    back = config_from_dict(json.loads(json.dumps(config_to_dict(c))))
    assert back == c
    assert isinstance(back.model.options, TTTOptions) and back.model.options.eta_max is None


def test_non_optional_field_rejects_null() -> None:
    d = config_to_dict(RunConfig(task="parity", model=ModelSpec("delta"), lr=1e-3, seed=0, steps=10))
    d["steps"] = None
    with pytest.raises(ValueError, match="steps"):
        config_from_dict(d)


def test_literal_field_rejects_unknown_member() -> None:
    d = config_to_dict(RunConfig(task="parity", model=ModelSpec("ttt"), lr=1e-3, seed=0, steps=10))
    model = d["model"]
    assert isinstance(model, dict)
    options = model["options"]
    assert isinstance(options, dict)
    options["inner_model"] = "transformer"
    with pytest.raises(ValueError, match="inner_model"):
        config_from_dict(d)
