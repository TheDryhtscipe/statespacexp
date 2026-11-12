"""Model-level tests: causality, shapes, parameter counts, gradient flow and the spectra
of the realised transitions."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from sst.models.mixers import HouseholderMixer
from sst.models import (
    MIXER_KINDS,
    DeltaProductOptions,
    MixerKind,
    MixerOptions,
    ModelSpec,
    TransformerOptions,
    TTTOptions,
    build_mixer,
    build_model,
    count_params,
)


def _small(kind: MixerKind, options: MixerOptions | None = None, **kw: bool) -> ModelSpec:
    return ModelSpec(kind, d_model=32, n_layers=2, n_heads=2, options=options, **kw)


VARIANTS: dict[str, ModelSpec] = {kind: _small(kind) for kind in MIXER_KINDS}
VARIANTS.update(
    {
        "deltaproduct_nh1": _small("deltaproduct", DeltaProductOptions(1)),
        "deltaproduct_nh4": _small("deltaproduct", DeltaProductOptions(4)),
        "deltaproduct_nh2_pos": _small("deltaproduct", DeltaProductOptions(2, allow_negative=False)),
        "ttt_linear_plain": _small("ttt", TTTOptions(inner_norm=False, normalise_keys=False)),
        "ttt_mlp_mb4": _small("ttt", TTTOptions(inner_model="mlp", minibatch=4)),
        "ttt_mlp_plain_mb3": _small("ttt", TTTOptions(inner_model="mlp", inner_norm=False, minibatch=3)),
        "transformer_nope": _small("transformer", TransformerOptions(pos="nope")),
        "delta_noconv_nonorm": _small("delta", use_short_conv=False, output_norm=False),
        "s4d_noconv": _small("s4d", use_short_conv=False),
    }
)

VOCAB, CLASSES = 6, 5


def _assert_causal(model: nn.Module, inputs: Tensor, positions: list[int]) -> None:
    with torch.no_grad():
        base: Tensor = model(inputs)
        for t in positions:
            changed = inputs.clone()
            changed[:, t] = (changed[:, t] + 1) % VOCAB
            out: Tensor = model(changed)
            torch.testing.assert_close(out[:, :t], base[:, :t], atol=1e-5, rtol=1e-5)
            # Non-vacuous: the change must reach the output at position t itself.
            assert (out[:, t] - base[:, t]).abs().max() > 1e-4


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_model_is_causal(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(VARIANTS[name], VOCAB, CLASSES).eval()
    inputs = torch.randint(0, VOCAB, (3, 14))
    _assert_causal(model, inputs, [0, 5, 13])


class _LeakyMixer(nn.Module):
    """Deliberately non-causal: every position sees the sum over the whole sequence."""

    def forward(self, x: Tensor) -> Tensor:
        return x + x.sum(dim=1, keepdim=True)


def test_causality_check_detects_a_leak() -> None:
    torch.manual_seed(0)
    model = build_model(_small("lin_attn"), VOCAB, CLASSES).eval()
    model.blocks[0].mixer = _LeakyMixer()
    with pytest.raises(AssertionError):
        _assert_causal(model, torch.randint(0, VOCAB, (3, 14)), [5])


@pytest.mark.parametrize("kind", MIXER_KINDS)
def test_default_shapes_and_param_counts(kind: MixerKind) -> None:
    torch.manual_seed(0)
    spec = ModelSpec(kind)
    model = build_model(spec, vocab_size=10, num_classes=7)
    out: Tensor = model(torch.randint(0, 10, (2, 16)))
    assert out.shape == (2, 16, 7)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    n = count_params(model)
    assert n == sum(p.numel() for p in model.parameters())
    print(f"{kind}: {n} params")


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_gradients_flow_to_every_parameter(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(VARIANTS[name], VOCAB, CLASSES)
    inputs = torch.randint(0, VOCAB, (4, 12))
    targets = torch.randint(0, CLASSES, (4, 12))
    logits: Tensor = model(inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, CLASSES), targets.reshape(-1))
    loss.backward()  # type: ignore[no-untyped-call]
    for pname, p in model.named_parameters():
        assert p.grad is not None, pname
        assert torch.isfinite(p.grad).all(), pname
        assert p.grad.abs().max() > 0, pname


def test_long_sequence_forward_is_finite() -> None:
    """lin_attn's state grows without bound; it must still not produce NaN at T = 512."""
    torch.manual_seed(0)
    for kind in ("lin_attn", "delta_neg", "ttt", "s4d"):
        model = build_model(_small(kind), VOCAB, CLASSES).eval()
        with torch.no_grad():
            out: Tensor = model(torch.randint(0, VOCAB, (2, 512)))
        assert torch.isfinite(out).all(), kind


def test_delta_mixer_equals_deltaproduct_mixer_with_one_step() -> None:
    torch.manual_seed(0)
    delta = build_mixer(_small("delta"))
    product = build_mixer(_small("deltaproduct", DeltaProductOptions(1, allow_negative=False)))
    product.load_state_dict(delta.state_dict())
    x = torch.randn(3, 20, 32)
    with torch.no_grad():
        torch.testing.assert_close(product(x), delta(x), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Spectra of the realised transitions A_t on random inputs
# ---------------------------------------------------------------------------


def _transitions(spec: ModelSpec, scale: float = 4.0, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    mixer = build_mixer(spec)
    x = torch.randn(4, 16, spec.d_model) * scale
    with torch.no_grad():
        return mixer.realised_transitions(x)


def _eig(a: Tensor) -> Tensor:
    eig: Tensor = torch.linalg.eigvals(a.double())
    return eig


def test_diag_pos_eigenvalues_in_open_unit_interval() -> None:
    eig = _eig(_transitions(_small("diag_pos")))
    assert eig.imag.abs().max() == 0
    assert eig.real.min() > 0 and eig.real.max() < 1


def test_diag_neg_eigenvalues_in_minus_one_one_and_reach_negative() -> None:
    eig = _eig(_transitions(_small("diag_neg")))
    assert eig.imag.abs().max() == 0
    assert eig.real.min() > -1 and eig.real.max() < 1
    assert eig.real.min() < -0.1


def test_diag_fixed_is_input_independent_in_minus_one_one() -> None:
    spec = _small("diag_fixed")
    a1 = _transitions(spec, seed=0)
    a2 = _transitions(spec, scale=10.0, seed=0)
    assert torch.equal(a1, a2)
    assert torch.equal(a1[0, :, 0], a1[3, :, 15])
    eig = _eig(a1)
    assert eig.real.min() > -1 and eig.real.max() < 1


def test_diag_gated_transitions_depend_on_input() -> None:
    a = _transitions(_small("diag_pos"))
    assert (a[0, :, 0] - a[0, :, 1]).abs().max() > 1e-3


def test_delta_eigenvalues_in_zero_one_so_no_sign_flip() -> None:
    """Negative control: with beta in (0, 1), I - beta k k^T has spectrum {1, 1 - beta}
    in [0, 1], so the transition can never flip a sign (needed for parity)."""
    a = _transitions(_small("delta")).double()
    eig = torch.linalg.eigvalsh(a)
    assert eig.min() >= -1e-6 and eig.max() <= 1 + 1e-6


def test_delta_neg_eigenvalues_in_minus_one_one_and_reach_negative() -> None:
    a = _transitions(_small("delta_neg")).double()
    eig = torch.linalg.eigvalsh(a)
    assert eig.min() > -1 and eig.max() <= 1 + 1e-6
    assert eig.min() < -0.1


@pytest.mark.parametrize("allow_negative", [False, True])
def test_deltaproduct_transitions_are_contractive(allow_negative: bool) -> None:
    eig = _eig(_transitions(_small("deltaproduct", DeltaProductOptions(3, allow_negative))))
    assert eig.abs().max() <= 1 + 1e-6
    if allow_negative:
        # A product of reflections reaches eigenvalues off the positive real axis.
        assert (eig.real < -0.1).any() or (eig.imag.abs() > 0.1).any()


def test_diag_complex_eigenvalues_non_real_and_inside_unit_disc() -> None:
    eig = _eig(_transitions(_small("diag_complex")))
    assert eig.abs().max() <= 1 + 1e-6
    assert (eig.imag.abs() > 1e-2).float().mean() > 0.5


def test_lin_attn_transition_is_identity() -> None:
    a = _transitions(_small("lin_attn"))
    assert torch.equal(a, torch.eye(16).expand_as(a))


def test_spec_rejects_mismatched_options() -> None:
    with pytest.raises(ValueError):
        ModelSpec("delta", options=TTTOptions())


# ---------------------------------------------------------------------------
# Learned initial state
# ---------------------------------------------------------------------------

STATEFUL_KINDS: tuple[MixerKind, ...] = (
    "lin_attn",
    "diag_fixed",
    "diag_pos",
    "diag_neg",
    "diag_complex",
    "delta",
    "delta_neg",
    "deltaproduct",
    "s4d",
    "lstm",
)


def _initial_state_params(mixer: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: p
        for name, p in mixer.named_parameters()
        if name.split(".")[-1].startswith("initial_")
    }


@pytest.mark.parametrize("kind", STATEFUL_KINDS)
def test_learned_initial_state_changes_the_output(kind: MixerKind) -> None:
    torch.manual_seed(0)
    mixer = build_mixer(_small(kind))
    params = _initial_state_params(mixer)
    assert params, f"{kind} has no learned initial state"
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        before: Tensor = mixer(x)
        for p in params.values():
            p.add_(torch.randn_like(p))
        after: Tensor = mixer(x)
    assert (after - before).abs().max() > 1e-3


@pytest.mark.parametrize("kind", STATEFUL_KINDS)
def test_initial_state_can_be_switched_off(kind: MixerKind) -> None:
    spec = _small(kind)
    assert getattr(spec, "learn_initial_state", False)
    off = ModelSpec(kind, d_model=32, n_layers=2, n_heads=2, learn_initial_state=False)
    assert not _initial_state_params(build_mixer(off))


def test_diag_neg_mixer_holds_parity_construction_with_initial_state() -> None:
    """The full diag_neg mixer (projections, short conv, core, gated output) tracks parity
    to length 512 when its weights are set by hand and S_0 = [[1, 0..], [-1, 0..], 0..].

    Token x in {0, 1} arrives as the one-hot vector e_x. alpha_proj gives pre-activation
    -30 (x = 0) or +30 (x = 1), so alpha' = 2 exp(-softplus(.)) - 1 is 1 or -1 to float32
    precision. Keys and values are 0 (no writes); queries and the gate are constant.
    """
    d = 4
    mixer = build_mixer(ModelSpec("diag_neg", d_model=d, n_layers=1, n_heads=1))
    params = dict(mixer.named_parameters())
    assert "initial_state" in params
    with torch.no_grad():
        for p in params.values():
            p.zero_()
        params["q_proj.weight"][:, :2] = 5.0
        for conv in ("q_conv", "k_conv", "v_conv"):
            params[f"{conv}.conv.weight"][..., -1] = 1.0  # identity convolution
        params["alpha_proj.weight"][:, 0] = -30.0
        params["alpha_proj.weight"][:, 1] = 30.0
        params["initial_state"][0, 0, 0] = 1.0
        params["initial_state"][0, 1, 0] = -1.0
        params["output.norm_weight"].fill_(1.0)
        params["output.gate_proj.weight"][:, :2] = 5.0
        params["output.out_proj.weight"].copy_(torch.eye(d))
        bits = torch.randint(0, 2, (16, 512), generator=torch.Generator().manual_seed(21))
        out: Tensor = mixer(nn.functional.one_hot(bits, d).float())
    parity = bits.cumsum(dim=1) % 2
    assert torch.equal(out[..., :2].argmax(dim=-1), parity)


# ---------------------------------------------------------------------------
# Matched initialisation of the negative-range variants
# ---------------------------------------------------------------------------


def test_diag_neg_starts_with_the_same_spectrum_as_diag_pos() -> None:
    """At zero input, diag_neg's alpha' = 2 alpha - 1 equals diag_pos's alpha (same seed),
    both uniform in [0.5, 0.99]: the pair differs in reachable range, not in init."""
    x = torch.zeros(1, 3, 32)
    spectra: list[Tensor] = []
    for kind in ("diag_pos", "diag_neg"):
        torch.manual_seed(0)
        mixer = build_mixer(_small(kind))
        with torch.no_grad():
            spectra.append(torch.diagonal(mixer.realised_transitions(x), dim1=-2, dim2=-1))
    pos, neg = spectra
    assert pos.min() >= 0.5 - 1e-6 and pos.max() <= 0.99 + 1e-6
    torch.testing.assert_close(neg, pos, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (_small("delta"), _small("delta_neg")),
        (
            _small("deltaproduct", DeltaProductOptions(2, allow_negative=False)),
            _small("deltaproduct", DeltaProductOptions(2, allow_negative=True)),
        ),
    ],
)
def test_negative_delta_starts_with_the_same_beta(positive: ModelSpec, negative: ModelSpec) -> None:
    """At zero input, beta (so the eigenvalue 1 - beta along k) is the same for the
    (0, 2) variant as for the (0, 1) one (same seed)."""
    x = torch.zeros(1, 3, 32)
    betas: list[Tensor] = []
    for spec in (positive, negative):
        torch.manual_seed(0)
        mixer = build_mixer(spec)
        assert isinstance(mixer, HouseholderMixer)
        with torch.no_grad():
            betas.append(mixer.beta(x))
    assert betas[0].min() > 0.0 and betas[0].max() < 1.0
    torch.testing.assert_close(betas[1], betas[0], atol=1e-6, rtol=0.0)


def test_s4d_runs_in_float64() -> None:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        model = build_model(_small("s4d"), VOCAB, CLASSES)
        assert all(p.dtype == torch.float64 for p in model.parameters())
        out: Tensor = model(torch.randint(0, VOCAB, (2, 10)))
    finally:
        torch.set_default_dtype(previous)
    assert out.dtype == torch.float64
