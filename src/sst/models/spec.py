"""Model specification: which sequence mixer, with which options, in the shared backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

MixerKind = Literal[
    "lin_attn",
    "diag_fixed",
    "diag_pos",
    "diag_neg",
    "diag_complex",
    "delta",
    "delta_neg",
    "deltaproduct",
    "ttt",
    "s4d",
    "transformer",
    "lstm",
]

MIXER_KINDS: tuple[MixerKind, ...] = get_args(MixerKind)


@dataclass(frozen=True)
class NoOptions:
    """For mixers with no options of their own."""


@dataclass(frozen=True)
class DeltaProductOptions:
    """n_householder steps per token; allow_negative: beta in (0, 2) if True, else (0, 1)."""

    n_householder: int = 2
    allow_negative: bool = True


@dataclass(frozen=True)
class TTTOptions:
    """Inner model, LN-residual inner model (Sun et al.), mini-batch size, eta_max for
    eta_t = eta_max * sigmoid(w . x_t), and L2 normalisation of keys and queries.

    eta_max=None means Sun et al.'s base rate for the inner model: 1.0 for linear and
    0.1 for mlp. The inner loss here is 1/2 ||f(k) - v||^2, so at the same eta the step
    is half the paper's (which has no 1/2).
    """

    inner_model: Literal["linear", "mlp"] = "linear"
    inner_norm: bool = True
    minibatch: int = 1
    eta_max: float | None = None
    normalise_keys: bool = True

    def resolved_eta_max(self) -> float:
        if self.eta_max is not None:
            return self.eta_max
        return 1.0 if self.inner_model == "linear" else 0.1


@dataclass(frozen=True)
class S4DOptions:
    """state_size N is the real state dimension per channel: N/2 complex conjugate pairs."""

    state_size: int = 32
    dt_min: float = 1e-3
    dt_max: float = 1e-1


@dataclass(frozen=True)
class TransformerOptions:
    pos: Literal["rope", "nope"] = "rope"


MixerOptions = NoOptions | DeltaProductOptions | TTTOptions | S4DOptions | TransformerOptions

_OPTION_TYPES: dict[MixerKind, type[MixerOptions]] = {
    "deltaproduct": DeltaProductOptions,
    "ttt": TTTOptions,
    "s4d": S4DOptions,
    "transformer": TransformerOptions,
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything that defines a model apart from vocabulary and class counts.

    options=None means the default options for the mixer kind. output_norm applies the
    per-head RMSNorm before the gated output projection of the recurrent mixers.
    learn_initial_state gives every recurrent mixer a learned initial state, shared
    across sequences and initialised N(0, 0.02): S_0 per head for the matrix-state
    mixers, x_0 for s4d, (h_0, c_0) for the LSTM. TTT always learns its W_0, so the
    flag does not change it; the transformer has no state.
    """

    mixer: MixerKind
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    use_short_conv: bool = True
    output_norm: bool = True
    learn_initial_state: bool = True
    options: MixerOptions | None = None

    def __post_init__(self) -> None:
        if self.mixer not in MIXER_KINDS:
            raise ValueError(f"unknown mixer kind {self.mixer!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        expected = _OPTION_TYPES.get(self.mixer, NoOptions)
        if self.options is not None and type(self.options) is not expected:
            raise ValueError(
                f"mixer {self.mixer!r} takes {expected.__name__}, got {type(self.options).__name__}"
            )
        if self.mixer == "diag_complex" and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("diag_complex needs an even head dimension")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def resolved_options(self) -> MixerOptions:
        if self.options is not None:
            return self.options
        return _OPTION_TYPES.get(self.mixer, NoOptions)()
