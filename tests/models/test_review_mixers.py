"""Independent review checks of the model zoo (bead statespacexp-96a027).

Each check uses its own reference, written from the mixer's name and definition rather
than from the code under test:

- every matrix-state mixer's forward pass equals an explicit matrix recurrence whose
  transitions A_t are built here from the definition the mixer's name promises, and the
  mixer's realised_transitions (used by the spectrum tests) are those same matrices;
- TTT-Linear (no LN, mini-batch 1, normalised keys, eta_max = 1) equals the delta mixer
  when the two share weights;
- the S4D layer equals a sequential complex recurrence with the learned initial state;
- mini-batch TTT equals Sun et al. (2024): within a mini-batch every gradient is taken
  at the weights from the start of the mini-batch, and token t reads
  W_t = W_{t'} - sum_{s=t'+1}^{t} eta_s grad l(W_{t'}; k_s, v_s);
- every mixer is causal at the evaluation length T = 512;
- the transition spectra stay in their closed ranges at saturating preactivations.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sst.models import (
    MIXER_KINDS,
    DeltaProductOptions,
    ModelSpec,
    TransformerOptions,
    TTTOptions,
    build_mixer,
    build_model,
)
from sst.models.mixers import (
    DiagComplexMixer,
    DiagFixedMixer,
    DiagGatedMixer,
    HouseholderMixer,
    RecurrentMixer,
    S4DMixer,
    TTTMixer,
    _merge_heads,
)
from sst.models.ttt import TTTInnerConfig, ttt_scan

D_MODEL, N_HEADS = 16, 2
HEAD = D_MODEL // N_HEADS


def _spec(
    kind: str, options: DeltaProductOptions | TTTOptions | None = None
) -> ModelSpec:
    spec = ModelSpec(kind, d_model=D_MODEL, n_heads=N_HEADS, options=options)  # type: ignore[arg-type]
    return spec


def _perturb(module: torch.nn.Module, scale: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


# ---------------------------------------------------------------------------
# Explicit matrix recurrence with transitions built from each mixer's definition
# ---------------------------------------------------------------------------

Step = tuple[Tensor, Tensor]  # (A [B, H, T, d, d], write [B, H, T, d, d])


def _outer(v: Tensor, k: Tensor) -> Tensor:
    return v.unsqueeze(-1) * k.unsqueeze(-2)


def _heads(y: Tensor, per_head: int) -> Tensor:
    b, t, _ = y.shape
    return y.view(b, t, N_HEADS, per_head).transpose(1, 2)


def _definition_steps(name: str, m: RecurrentMixer, x: Tensor) -> list[Step]:
    """The per-token steps S <- S A + W that the mixer's name promises."""
    b, t, _ = x.shape
    eye = torch.eye(HEAD, dtype=x.dtype)
    _, k, v = m.project(x)
    if name == "lin_attn":
        return [
            (
                eye.expand(b, N_HEADS, t, HEAD, HEAD),
                _outer(v[:, :, :, 0], k[:, :, :, 0]),
            )
        ]
    if name == "diag_fixed":
        assert isinstance(m, DiagFixedMixer)
        a = torch.tanh(m.alpha_raw)[None, :, None, :].expand(b, N_HEADS, t, HEAD)
        return [(torch.diag_embed(a), _outer(v[:, :, :, 0], k[:, :, :, 0]))]
    if name in ("diag_pos", "diag_neg"):
        assert isinstance(m, DiagGatedMixer)
        alpha = torch.sigmoid(
            -_heads(m.alpha_proj(x), HEAD)
        )  # exp(-softplus(z)) in (0, 1)
        if name == "diag_neg":
            alpha = 2.0 * alpha - 1.0
        return [(torch.diag_embed(alpha), _outer(v[:, :, :, 0], k[:, :, :, 0]))]
    if name == "diag_complex":
        assert isinstance(m, DiagComplexMixer)
        r = torch.sigmoid(-_heads(m.radius_proj(x), HEAD // 2))
        th = _heads(m.angle_proj(x), HEAD // 2)
        a = x.new_zeros(b, N_HEADS, t, HEAD, HEAD)
        for i in range(HEAD // 2):
            # Column pair (2i, 2i+1) of S is one complex number, multiplied by r e^{i th}.
            c, s = r[..., i] * torch.cos(th[..., i]), r[..., i] * torch.sin(th[..., i])
            a[..., 2 * i, 2 * i] = c
            a[..., 2 * i, 2 * i + 1] = s
            a[..., 2 * i + 1, 2 * i] = -s
            a[..., 2 * i + 1, 2 * i + 1] = c
        return [(a, _outer(v[:, :, :, 0], k[:, :, :, 0]))]
    assert isinstance(m, HouseholderMixer)
    beta_max = {"delta": 1.0, "delta_neg": 2.0, "dp_nh2_neg": 2.0, "dp_nh3_pos": 1.0}[
        name
    ]
    n = k.shape[3]
    beta = beta_max * torch.sigmoid(m.beta_proj(x)).view(b, t, n, N_HEADS).permute(
        0, 3, 1, 2
    )
    steps: list[Step] = []
    for j in range(n):
        k_j, v_j, b_j = k[:, :, :, j], v[:, :, :, j], beta[..., j, None, None]
        torch.testing.assert_close(k_j.norm(dim=-1), torch.ones_like(k_j[..., 0]))
        steps.append((eye - b_j * _outer(k_j, k_j), b_j * _outer(v_j, k_j)))
    return steps


MATRIX_MIXERS: dict[str, ModelSpec] = {
    "lin_attn": _spec("lin_attn"),
    "diag_fixed": _spec("diag_fixed"),
    "diag_pos": _spec("diag_pos"),
    "diag_neg": _spec("diag_neg"),
    "diag_complex": _spec("diag_complex"),
    "delta": _spec("delta"),
    "delta_neg": _spec("delta_neg"),
    "dp_nh2_neg": _spec("deltaproduct", DeltaProductOptions(2, allow_negative=True)),
    "dp_nh3_pos": _spec("deltaproduct", DeltaProductOptions(3, allow_negative=False)),
}


def _explicit(name: str, m: RecurrentMixer, x: Tensor) -> tuple[Tensor, Tensor]:
    """(forward output, transition products) of the explicit recurrence S <- S A + W."""
    steps = _definition_steps(name, m, x)
    q, _, _ = m.project(x)
    b, _, t, _ = q.shape
    s0 = m.initial_state
    assert s0 is not None
    state = s0.unsqueeze(0).expand(b, N_HEADS, HEAD, HEAD)
    outs: list[Tensor] = []
    products: list[Tensor] = []
    for i in range(t):
        total = torch.eye(HEAD, dtype=x.dtype).expand(b, N_HEADS, HEAD, HEAD)
        for a, w in steps:
            state = state @ a[:, :, i] + w[:, :, i]
            total = total @ a[:, :, i]
        products.append(total)
        outs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, :, i]))
    out: Tensor = m.output(_merge_heads(torch.stack(outs, dim=2)), x)
    return out, torch.stack(products, dim=2)


def _setup(name: str) -> tuple[RecurrentMixer, Tensor]:
    torch.manual_seed(0)
    m = build_mixer(MATRIX_MIXERS[name]).double()
    assert isinstance(m, RecurrentMixer)
    _perturb(m, 0.3, seed=1)
    return m, torch.randn(2, 24, D_MODEL, dtype=torch.float64)


@pytest.mark.parametrize("name", sorted(MATRIX_MIXERS))
def test_mixer_equals_explicit_recurrence_of_its_definition(name: str) -> None:
    m, x = _setup(name)
    with torch.no_grad():
        expected, products = _explicit(name, m, x)
        got = m(x)
        realised = m.realised_transitions(x)
    assert expected.abs().max() > 0.1
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(realised, products, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("name", "attr", "value"),
    [("diag_pos", "negative", True), ("delta", "beta_max", 2.0)],
)
def test_explicit_recurrence_check_catches_a_misnamed_transition(
    name: str, attr: str, value: bool | float
) -> None:
    """diag_pos with the diag_neg map, or delta with beta in (0, 2), fails the check."""
    m, x = _setup(name)
    setattr(m, attr, value)
    with torch.no_grad():
        expected, _ = _explicit(name, m, x)
        got = m(x)
    assert (got - expected).abs().max() > 1e-3


# ---------------------------------------------------------------------------
# TTT-Linear online = delta rule, at the mixer level
# ---------------------------------------------------------------------------


def test_ttt_linear_mixer_equals_delta_mixer_with_shared_weights() -> None:
    torch.manual_seed(0)
    delta = build_mixer(_spec("delta"))
    ttt = build_mixer(
        _spec(
            "ttt",
            TTTOptions(
                inner_model="linear",
                inner_norm=False,
                minibatch=1,
                eta_max=1.0,
                normalise_keys=True,
            ),
        )
    )
    assert isinstance(delta, HouseholderMixer) and isinstance(ttt, TTTMixer)
    _perturb(delta, 0.3, seed=2)
    with torch.no_grad():
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv",
            "k_conv",
            "v_conv",
            "output",
        ):
            getattr(ttt, name).load_state_dict(getattr(delta, name).state_dict())
        ttt.eta_proj.load_state_dict(delta.beta_proj.state_dict())
        assert delta.initial_state is not None
        ttt.inner_init[0].copy_(delta.initial_state)
        x = torch.randn(2, 40, D_MODEL)
        expected = delta(x)
        got = ttt(x)
        ttt.inner_init[0].mul_(-1.0)
        changed = ttt(x)
    assert expected.abs().max() > 0.1
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    assert (changed - expected).abs().max() > 1e-3


# ---------------------------------------------------------------------------
# S4D against a sequential complex recurrence
# ---------------------------------------------------------------------------


def test_s4d_layer_equals_sequential_recurrence_with_initial_state() -> None:
    torch.manual_seed(0)
    mixer = build_mixer(_spec("s4d"))
    assert isinstance(mixer, S4DMixer)
    ssm = mixer.ssm.double()
    _perturb(ssm, 0.1, seed=3)
    assert ssm.initial_state_re is not None and ssm.initial_state_im is not None
    u = torch.randn(2, 200, D_MODEL, dtype=torch.float64)
    with torch.no_grad():
        a = torch.complex(-torch.exp(ssm.a_re_log), ssm.a_im)
        lam = torch.exp(torch.exp(ssm.log_dt)[:, None] * a)
        b_bar = (lam - 1.0) / a * torch.complex(ssm.b_re, ssm.b_im)
        c = torch.complex(ssm.c_re, ssm.c_im)
        state = torch.complex(ssm.initial_state_re, ssm.initial_state_im).expand(
            2, -1, -1
        )
        ys: list[Tensor] = []
        for t in range(u.shape[1]):
            state = lam * state + b_bar * u[:, t, :, None]
            ys.append(2.0 * (c * state).sum(dim=-1).real + ssm.skip * u[:, t])
        expected = torch.stack(ys, dim=1)
        got = ssm(u)
    assert lam.abs().max() < 1.0
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Mini-batch TTT against Sun et al. (2024)
# ---------------------------------------------------------------------------


def _inner_f(
    weights: list[Tensor], x: Tensor, mlp: bool, ln: tuple[Tensor, Tensor] | None
) -> Tensor:
    if mlp:
        hidden = F.gelu(torch.einsum("bhoi,bhi->bho", weights[0], x) + weights[1])
        y = torch.einsum("bhoi,bhi->bho", weights[2], hidden) + weights[3]
    else:
        y = torch.einsum("bhoi,bhi->bho", weights[0], x)
    if ln is None:
        return y
    return x + F.layer_norm(y, (y.shape[-1],), eps=1e-5) * ln[0] + ln[1]


def _sun_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    eta: Tensor,
    init: tuple[Tensor, ...],
    mlp: bool,
    ln: tuple[Tensor, Tensor] | None,
    minibatch: int,
) -> Tensor:
    b, h, t, _ = q.shape
    weights = [p.unsqueeze(0).expand(b, *p.shape).clone() for p in init]
    outs: list[Tensor] = []
    for start in range(0, t, minibatch):
        at_start = [w.detach().requires_grad_(True) for w in weights]
        total = [torch.zeros_like(w) for w in weights]
        current = weights
        for s in range(start, min(start + minibatch, t)):
            loss = (
                0.5
                * (_inner_f(at_start, k[:, :, s], mlp, ln) - v[:, :, s]).pow(2).sum()
            )
            grads = torch.autograd.grad(loss, at_start)
            total = [
                acc + eta[:, :, s].view(b, h, *([1] * (g.dim() - 2))) * g
                for acc, g in zip(total, grads, strict=True)
            ]
            current = [w.detach() - acc for w, acc in zip(at_start, total, strict=True)]
            outs.append(_inner_f(current, q[:, :, s], mlp, ln).detach())
        weights = current
    return torch.stack(outs, dim=2)


@pytest.mark.parametrize("mlp", [False, True])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.parametrize("minibatch", [1, 5, 16])
def test_ttt_scan_matches_sun_et_al_minibatch_definition(
    mlp: bool, use_norm: bool, minibatch: int
) -> None:
    g = torch.Generator().manual_seed(4)
    b, h, t, d = 2, 2, 37, 6  # 37: the last mini-batch is partial
    q = F.normalize(torch.randn(b, h, t, d, generator=g, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(b, h, t, d, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, t, d, generator=g, dtype=torch.float64)
    eta = 0.1 * torch.rand(b, h, t, generator=g, dtype=torch.float64)
    shapes = [(h, 4 * d, d), (h, 4 * d), (h, d, 4 * d), (h, d)] if mlp else [(h, d, d)]
    init = tuple(
        0.3 * torch.randn(*s, generator=g, dtype=torch.float64) for s in shapes
    )
    ln = (
        (
            1.0 + 0.2 * torch.randn(h, d, generator=g, dtype=torch.float64),
            0.1 * torch.randn(h, d, generator=g, dtype=torch.float64),
        )
        if use_norm
        else None
    )
    cfg = TTTInnerConfig("mlp" if mlp else "linear", use_norm, minibatch)
    expected = _sun_reference(
        q,
        k,
        v,
        eta,
        init,
        mlp,
        None if ln is None else (ln[0][None], ln[1][None]),
        minibatch,
    )
    with torch.no_grad():
        dual = ttt_scan(q, k, v, eta, init, cfg, ln).outputs
        primal = ttt_scan(q, k, v, eta, init, cfg, ln, return_states=True).outputs
    assert expected.abs().max() > 0.1
    torch.testing.assert_close(dual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(primal, expected, atol=1e-10, rtol=1e-10)
    if minibatch > 1:
        # Non-vacuous: fully online TTT (gradients at the current weights) differs.
        online = ttt_scan(
            q, k, v, eta, init, TTTInnerConfig(cfg.inner_model, use_norm, 1), ln
        )
        assert (online.outputs - expected).abs().max() > 1e-6


# ---------------------------------------------------------------------------
# Causality at the evaluation length
# ---------------------------------------------------------------------------

CAUSAL_VARIANTS: dict[str, ModelSpec] = {kind: _spec(kind) for kind in MIXER_KINDS}
CAUSAL_VARIANTS.update(
    {
        "ttt_mlp_mb16": _spec("ttt", TTTOptions(inner_model="mlp", minibatch=16)),
        "ttt_linear_mb8": _spec("ttt", TTTOptions(minibatch=8)),
        "transformer_nope": ModelSpec(
            "transformer",
            d_model=D_MODEL,
            n_heads=N_HEADS,
            options=TransformerOptions(pos="nope"),
        ),
    }
)


@pytest.mark.parametrize("name", sorted(CAUSAL_VARIANTS))
def test_causal_at_length_512(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(CAUSAL_VARIANTS[name], 6, 5).double().eval()
    inputs = torch.randint(0, 6, (1, 512))
    with torch.no_grad():
        base: Tensor = model(inputs)
        for pos in (64, 300, 511):
            changed = inputs.clone()
            changed[:, pos] = (changed[:, pos] + 1) % 6
            out: Tensor = model(changed)
            assert (out[:, :pos] - base[:, :pos]).abs().max() < 1e-9, pos
            assert (out[:, pos] - base[:, pos]).abs().max() > 1e-4, pos


# ---------------------------------------------------------------------------
# Spectra at saturating preactivations (float32, as trained)
# ---------------------------------------------------------------------------


def _saturated(spec: ModelSpec) -> tuple[RecurrentMixer, Tensor]:
    torch.manual_seed(0)
    m = build_mixer(spec)
    assert isinstance(m, RecurrentMixer)
    _perturb(m, 1.0, seed=5)
    return m, torch.randn(4, 64, spec.d_model) * 100.0


def test_diag_pos_never_negative_and_diag_neg_closed_at_saturation() -> None:
    m_pos, x = _saturated(_spec("diag_pos"))
    m_neg, _ = _saturated(_spec("diag_neg"))
    with torch.no_grad():
        pos = torch.diagonal(m_pos.realised_transitions(x), dim1=-2, dim2=-1)
        neg = torch.diagonal(m_neg.realised_transitions(x), dim1=-2, dim2=-1)
    assert pos.min() >= 0.0 and pos.max() <= 1.0
    assert neg.min() >= -1.0 and neg.max() <= 1.0
    assert neg.min() < -0.99  # the negative end is reachable


@pytest.mark.parametrize(("kind", "lowest"), [("delta", 0.0), ("delta_neg", -1.0)])
def test_delta_eigenvalue_along_key_stays_in_range_at_saturation(
    kind: str, lowest: float
) -> None:
    m, x = _saturated(_spec(kind))
    assert isinstance(m, HouseholderMixer)
    with torch.no_grad():
        _, k, _ = m.project(x)
        along_k = 1.0 - m.beta(x).squeeze(-1) * k.squeeze(3).pow(2).sum(dim=-1)
    # float32 rounding of ||k|| = 1 and beta = beta_max allows a few ulps past the bound.
    assert along_k.min() >= lowest - 1e-5
    assert along_k.max() <= 1.0 + 1e-6
    assert along_k.min() < lowest + 0.01  # the bound is reachable


def test_positive_deltaproduct_two_steps_has_real_nonnegative_spectrum() -> None:
    """A product of two PSD matrices (I - beta k k^T, beta in (0, 1)) has real eigenvalues
    in [0, 1], so deltaproduct with allow_negative=False cannot rotate or flip a sign."""
    torch.manual_seed(0)
    m = build_mixer(
        _spec("deltaproduct", DeltaProductOptions(2, allow_negative=False))
    ).double()
    _perturb(m, 1.0, seed=6)
    with torch.no_grad():
        eig = torch.linalg.eigvals(
            m.realised_transitions(torch.randn(4, 64, D_MODEL).double() * 3)
        )
    assert eig.imag.abs().max() < 1e-6
    assert eig.real.min() > -1e-6 and eig.real.max() < 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Known failure: TTT-MLP without inner LN diverges with mini-batches at the default eta_max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("minibatch", [4, 16])
def test_ttt_mlp_without_inner_norm_is_finite_at_length_512(minibatch: int) -> None:
    """TTT-MLP with inner_norm=False and minibatch > 1 is a supported configuration, but at
    the default eta_max = 1.0 the inner SGD diverges and the logits become NaN, at
    initialisation and at default sizes (Sun et al. use eta_base = 0.1 for TTT-MLP;
    eta_max = 0.1 stays finite here). A NaN at T = 512 would count as a failure of the
    family rather than of its step size."""
    torch.manual_seed(0)
    spec = ModelSpec(
        "ttt",
        options=TTTOptions(inner_model="mlp", inner_norm=False, minibatch=minibatch),
    )
    model = build_model(spec, 4, 2).eval()
    with torch.no_grad():
        out: Tensor = model(torch.randint(0, 4, (2, 512)))
    assert torch.isfinite(out).all()
