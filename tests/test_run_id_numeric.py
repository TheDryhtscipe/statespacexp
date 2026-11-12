"""Review finding: an int in a float field gives a run_id that changes after a JSON round trip."""

import json
import time
from pathlib import Path

import pytest

from sst.config import RunConfig, config_from_dict, config_to_dict
from sst.models import ModelSpec
from sst.models.spec import S4DOptions, TTTOptions
from sst.queue import JobQueue, load_job


def _round_trip(c: RunConfig) -> RunConfig:
    return config_from_dict(json.loads(json.dumps(config_to_dict(c))))


@pytest.mark.parametrize(
    "cfg",
    [
        RunConfig("s5", ModelSpec("lstm"), lr=1e-3, seed=0, steps=10, weight_decay=0),
        RunConfig("s5", ModelSpec("lstm"), lr=1, seed=0, steps=10),
        RunConfig("s5", ModelSpec("ttt", options=TTTOptions(eta_max=1)), lr=1e-3, seed=0, steps=10),
        RunConfig("s5", ModelSpec("s4d", options=S4DOptions(dt_max=1)), lr=1e-3, seed=0, steps=10),
    ],
)
def test_run_id_survives_json_round_trip(cfg: RunConfig) -> None:
    assert _round_trip(cfg).run_id == cfg.run_id


def test_numerically_equal_configs_share_a_run_id() -> None:
    a = RunConfig("s5", ModelSpec("lstm"), lr=1e-3, seed=0, steps=10, weight_decay=0)
    b = RunConfig("s5", ModelSpec("lstm"), lr=1e-3, seed=0, steps=10, weight_decay=0.0)
    assert a == b  # dataclass equality already says these are the same run
    assert a.run_id == b.run_id


def test_queued_job_file_names_the_run_id_the_child_will_record(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    q.submit([RunConfig("s5", ModelSpec("lstm"), lr=1e-3, seed=0, steps=10, weight_decay=0)])
    (path,) = list(q.pending.glob("*.json"))
    assert load_job(path).run_id == path.stem.rsplit("-", 1)[-1]


def test_negative_zero_shares_a_run_id() -> None:
    a = RunConfig("s5", ModelSpec("lstm"), lr=1e-3, seed=0, steps=10, weight_decay=0.0)
    b = RunConfig("s5", ModelSpec("lstm"), lr=1e-3, seed=0, steps=10, weight_decay=-0.0)
    assert a == b
    assert a.run_id == b.run_id


def test_config_json_round_trip_is_cheap() -> None:
    """Analysis recomputes ids for thousands of records; field-type lookup must be cached."""
    c = RunConfig("parity", ModelSpec("ttt"), lr=1e-3, seed=0, steps=10)
    t = time.perf_counter()
    for _ in range(1000):
        config_from_dict(json.loads(json.dumps(config_to_dict(c))))
    assert time.perf_counter() - t < 1.0


def test_graphed_config_refuses_a_non_cuda_device() -> None:
    """cuda_graphs is in run_id, so an eager CPU run must not record under the graphed id."""
    from sst.train import train

    c = RunConfig("parity", ModelSpec("lstm"), lr=1e-3, seed=0, steps=2, cuda_graphs=True)
    with pytest.raises(ValueError, match="cuda_graphs"):
        train(c, device="cpu")


def test_negative_zero_eta_max_survives_round_trip() -> None:
    c = RunConfig("parity", ModelSpec("ttt", options=TTTOptions(eta_max=-0.0)), lr=1e-3, seed=0, steps=10)
    assert config_from_dict(config_to_dict(c)).run_id == c.run_id


def test_field_type_cache_tells_same_named_classes_apart() -> None:
    import dataclasses as dc

    from sst.config import declared_scalar

    def make(kind: type) -> type:
        @dc.dataclass
        class C:
            x: int = 0

        C.__annotations__["x"] = kind
        return C

    a, b = make(int), make(float)
    da, db = declared_scalar(a, "x"), declared_scalar(b, "x")
    assert da is not None and da.kind is int
    assert db is not None and db.kind is float
