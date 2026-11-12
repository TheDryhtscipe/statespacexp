"""Run configuration: everything that defines one training run, and its stable ``run_id``.

This module does not import torch, so the queue and planning commands start fast.

run_id
------
``run_id`` is the first 16 hex digits of SHA-256 over the canonical JSON of the
config's *scientific* fields: every field except those in
:data:`EXECUTION_ONLY_FIELDS`, which change how a run executes but not what it
computes (``eval_batch_size`` changes nothing beyond rounding at argmax ties, see
:func:`sst.evaluation.evaluate`). ``cuda_graphs`` is hashed: graph capture changes
float rounding, and training amplifies it, so graphed and eager runs differ. The config is normalised
before hashing: the model's options are resolved to their concrete defaults,
and the default eval sequence counts are filled in, so two spellings of the
same run get the same id. ``protocol_version`` is in the hash; bump it when a
code change alters what a config means (for example the data stream or the
optimiser), so old results are not mistaken for new ones.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, fields

from sst.evaluation import JSONValue
from sst.models import ModelSpec
from sst.models.spec import MIXER_KINDS, MixerOptions, TTTOptions
from sst.tasks import TASK_NAMES, FlipFlopTask, get_task

PROTOCOL_VERSION = 1

#: Fields that do not enter ``run_id``.
EXECUTION_ONLY_FIELDS: frozenset[str] = frozenset({"eval_batch_size"})

#: Sentinel for "use the task's default sequence count".
TASK_DEFAULT = -1

GROUP_EVAL_SEQUENCES = 1024
FLIPFLOP_EVAL_SEQUENCES = 4096


def default_eval_sequences(task: str) -> int:
    """1024 for group tasks; 4096 for flip-flop tasks, whose reads are only ~45% of positions."""
    return FLIPFLOP_EVAL_SEQUENCES if isinstance(get_task(task), FlipFlopTask) else GROUP_EVAL_SEQUENCES


@dataclass(frozen=True)
class RunConfig:
    """One training run. See the module docstring for ``run_id``.

    ``val_sequences`` / ``eval_sequences`` of ``TASK_DEFAULT`` resolve to
    :func:`default_eval_sequences` (read them through ``n_val_sequences`` and
    ``n_eval_sequences``). The sentinel stays stored, so ``dataclasses.replace``
    with a new task picks up that task's default. Model options of ``None`` are
    replaced by the mixer's default options object. ``data_chunk_steps`` is part of the data
    definition: training batches are drawn ``data_chunk_steps`` at a time with a
    single ``task.sample`` call (see :mod:`sst.train`).
    """

    task: str
    model: ModelSpec
    lr: float
    seed: int
    steps: int
    batch_size: int = 128
    train_length: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_sequences: int = TASK_DEFAULT
    eval_sequences: int = TASK_DEFAULT
    eval_seed: int = 0
    eval_multiplier: int = 8
    log_every: int = 100
    val_every: int = 1000
    curve_val_sequences: int = 512
    data_chunk_steps: int = 50
    protocol_version: int = PROTOCOL_VERSION
    cuda_graphs: bool = False
    eval_batch_size: int = 128

    def __post_init__(self) -> None:
        if self.task not in TASK_NAMES:
            raise ValueError(f"unknown task {self.task!r}; expected one of {TASK_NAMES}")
        for name in ("steps", "batch_size", "train_length", "log_every", "val_every",
                     "curve_val_sequences", "data_chunk_steps", "eval_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.seed < 0 or self.eval_seed < 0 or self.warmup_steps < 0:
            raise ValueError("seed, eval_seed and warmup_steps must be >= 0")
        if not self.lr > 0:
            raise ValueError("lr must be > 0")
        for name in ("val_sequences", "eval_sequences"):
            value: int = getattr(self, name)
            if value != TASK_DEFAULT and value < 1:
                raise ValueError(f"{name} must be >= 1 or TASK_DEFAULT")
        if self.model.options is None:
            object.__setattr__(self, "model", dataclasses.replace(self.model, options=self.model.resolved_options()))

    @property
    def n_val_sequences(self) -> int:
        return default_eval_sequences(self.task) if self.val_sequences == TASK_DEFAULT else self.val_sequences

    @property
    def n_eval_sequences(self) -> int:
        return default_eval_sequences(self.task) if self.eval_sequences == TASK_DEFAULT else self.eval_sequences

    def _resolved_dict(self) -> dict[str, JSONValue]:
        d = config_to_dict(self)
        options = self.model.options
        model_d = d["model"]
        if isinstance(options, TTTOptions) and isinstance(model_d, dict):
            options_d = model_d["options"]
            if isinstance(options_d, dict):
                # eta_max=None means the inner model's base rate; hash the rate itself.
                options_d["eta_max"] = float(options.resolved_eta_max())
        d["val_sequences"] = self.n_val_sequences
        d["eval_sequences"] = self.n_eval_sequences
        for name in EXECUTION_ONLY_FIELDS:
            del d[name]
        return _without_negative_zero(d)

    @property
    def run_id(self) -> str:
        d = self._resolved_dict()
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def cell_key(self) -> str:
        """Everything but ``lr`` and ``seed`` (and execution-only fields), as canonical JSON.

        Runs with the same cell key belong to one (family, task, depth) cell under one
        protocol; the LR sweep and the seed runs of a cell differ only in lr and seed.
        """
        d = self._resolved_dict()
        del d["lr"], d["seed"]
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------ JSON


def _without_negative_zero(d: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Map -0.0 to 0.0 throughout, so numerically equal configs hash alike."""
    out: dict[str, JSONValue] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _without_negative_zero(v)
        elif isinstance(v, float):
            out[k] = v + 0.0
        else:
            out[k] = v
    return out


def _scalar_fields(obj: object) -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {}
    for f in fields(obj):  # type: ignore[arg-type]
        v = getattr(obj, f.name)
        if v is None or isinstance(v, (bool, int, float, str)):
            declared = declared_scalar(type(obj), f.name)
            if declared is not None and declared.kind is float and isinstance(v, int) and not isinstance(v, bool):
                # Encode an int in a float field as a float, so run_id survives a JSON round trip.
                v = float(v)
            if isinstance(v, float):
                v = v + 0.0  # -0.0 == 0.0, so encode both as 0.0
            out[f.name] = v
    return out


_SCALAR_TYPES: tuple[type, ...] = (bool, int, float, str)


@dataclass(frozen=True)
class ScalarField:
    kind: type
    optional: bool
    allowed: tuple[str, ...] | None  # the members of a Literal[...] field


def declared_scalar(cls: type, name: str) -> ScalarField | None:
    """The declared scalar type of field ``name``; None if it is not a scalar field."""
    key = (cls, name)
    if key not in _DECLARED_CACHE:
        _DECLARED_CACHE[key] = _declared_scalar_uncached(cls, name)
    return _DECLARED_CACHE[key]


#: get_type_hints is slow, and analysis encodes thousands of configs.
_DECLARED_CACHE: dict[tuple[type, str], ScalarField | None] = {}


def _declared_scalar_uncached(cls: type, name: str) -> ScalarField | None:
    hint = typing.get_type_hints(cls)[name]
    args = typing.get_args(hint) if isinstance(hint, types.UnionType) else (hint,)
    optional = type(None) in args
    rest = [a for a in args if a is not type(None)]
    if len(rest) != 1:
        return None
    (base,) = rest
    if typing.get_origin(base) is typing.Literal:
        members = typing.get_args(base)
        if members and all(isinstance(m, str) for m in members):
            return ScalarField(str, optional, tuple(str(m) for m in members))
        return None
    if base in _SCALAR_TYPES:
        return ScalarField(base, optional, None)
    return None


def spec_to_dict(spec: ModelSpec) -> dict[str, JSONValue]:
    d = _scalar_fields(spec)
    d["options"] = _scalar_fields(spec.resolved_options())
    return d


def config_to_dict(c: RunConfig) -> dict[str, JSONValue]:
    d = _scalar_fields(c)
    d["model"] = spec_to_dict(c.model)
    return d


def _coerce(value: JSONValue, like: bool | int | float | str, name: str) -> bool | int | float | str:
    """Check ``value`` has the type of the default ``like``; ints are accepted for floats."""
    if isinstance(like, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(like, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif isinstance(like, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(value, str):
        return value
    raise ValueError(f"field {name!r}: expected {type(like).__name__}, got {value!r}")


def replace_fields[T](template: T, d: dict[str, JSONValue], skip: frozenset[str] = frozenset()) -> T:
    changes: dict[str, bool | int | float | str | None] = {}
    for f in fields(template):  # type: ignore[arg-type]
        if f.name in skip or f.name not in d:
            continue
        declared = declared_scalar(type(template), f.name)
        if declared is None:
            raise ValueError(f"field {f.name!r} is not a scalar field and cannot be read from JSON")
        value = d[f.name]
        if value is None:
            if not declared.optional:
                raise ValueError(f"field {f.name!r}: expected {declared.kind.__name__}, got None")
            changes[f.name] = None
            continue
        like: bool | int | float | str = declared.kind()
        coerced = _coerce(value, like, f.name)
        if declared.allowed is not None and coerced not in declared.allowed:
            raise ValueError(f"field {f.name!r}: expected one of {list(declared.allowed)}, got {value!r}")
        changes[f.name] = coerced
    unknown = set(d) - {f.name for f in fields(template)} - skip  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    return dataclasses.replace(template, **changes)  # type: ignore[type-var]


def spec_from_dict(d: dict[str, JSONValue]) -> ModelSpec:
    mixer = d.get("mixer")
    if not isinstance(mixer, str):
        raise ValueError(f"model.mixer must be a string, got {mixer!r}")
    kind = next((k for k in MIXER_KINDS if k == mixer), None)
    if kind is None:
        raise ValueError(f"unknown mixer {mixer!r}")
    base = ModelSpec(kind)
    raw_options = d.get("options")
    options: MixerOptions = base.resolved_options()
    if raw_options is not None:
        if not isinstance(raw_options, dict):
            raise ValueError("model.options must be an object")
        options = replace_fields(options, raw_options)
    spec = replace_fields(base, d, skip=frozenset({"mixer", "options"}))
    return dataclasses.replace(spec, options=options)


def config_from_dict(d: dict[str, JSONValue]) -> RunConfig:
    task = d.get("task")
    model = d.get("model")
    if not isinstance(task, str) or not isinstance(model, dict):
        raise ValueError("config needs a string 'task' and an object 'model'")
    template = RunConfig(task=task, model=spec_from_dict(model), lr=1.0, seed=0, steps=1)
    return replace_fields(template, d, skip=frozenset({"model"}))


def spec_label(spec: ModelSpec) -> str:
    """Readable family name: the mixer kind plus every non-default field except n_layers."""
    parts: list[str] = []
    base = ModelSpec(spec.mixer)
    for f in fields(spec):
        if f.name in ("mixer", "n_layers", "options"):
            continue
        if getattr(spec, f.name) != getattr(base, f.name):
            parts.append(f"{f.name}={getattr(spec, f.name)}")
    default_opts = base.resolved_options()
    opts = spec.resolved_options()
    for f in fields(opts):
        if getattr(opts, f.name) != getattr(default_opts, f.name):
            parts.append(f"{f.name}={getattr(opts, f.name)}")
    return spec.mixer + (f"[{','.join(parts)}]" if parts else "")


def family_key(spec: ModelSpec) -> str:
    """Canonical JSON of the spec without n_layers: identifies a model family."""
    d = spec_to_dict(spec)
    del d["n_layers"]
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
