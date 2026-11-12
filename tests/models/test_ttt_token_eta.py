"""TTTOptions.token_eta: the published in-mini-batch rate scaling (test-time-training/
ttt-lm-pytorch, ttt.py). The output at position i (1-based) of a mini-batch reads

    W_i = W_start - s_i * sum_{r <= i} eta_r grad l(W_start; k_r, v_r),
    s_i = max(1 / i + offset_i, 0),

and the next mini-batch starts from W at the chunk's last position (s_c for a chunk of
c tokens, so a short final chunk uses s_c, not s_minibatch).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from sst.models import ModelSpec, TTTOptions, build_mixer
from sst.models.mixers import TTTMixer
from sst.models.ttt import InnerModel, TTTInnerConfig, inner_forward, ttt_scan


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _l2(x: Tensor) -> Tensor:
    unit: Tensor = x / x.norm(dim=-1, keepdim=True)
    return unit


def _init(model: InnerModel, h: int, d: int, seed: int) -> tuple[Tensor, ...]:
    g = _gen(seed)
    if model == "linear":
        return (torch.randn(h, d, d, generator=g) * 0.5,)
    return (
        torch.randn(h, 4 * d, d, generator=g) * 0.5,
        torch.randn(h, 4 * d, generator=g) * 0.5,
        torch.randn(h, d, 4 * d, generator=g) * 0.3,
        torch.randn(h, d, generator=g) * 0.5,
    )


def _reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    eta: Tensor,
    init: tuple[Tensor, ...],
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
    scale: Tensor,
) -> Tensor:
    """Token by token, autograd gradients, published row scaling s_i on the readout."""
    b, h, t, _ = q.shape
    start = tuple(p.unsqueeze(0).unsqueeze(2).expand(b, h, 1, *p.shape[1:]).clone() for p in init)
    outs: list[Tensor] = []
    for first in range(0, t, cfg.minibatch):
        acc = tuple(torch.zeros_like(p) for p in start)
        current = start
        for s in range(first, min(first + cfg.minibatch, t)):
            leaf = tuple(p.detach().requires_grad_(True) for p in start)
            ks, vs = k[:, :, s : s + 1], v[:, :, s : s + 1]
            loss = 0.5 * (inner_forward(leaf, ks, cfg, norm) - vs).pow(2).sum()
            grads = torch.autograd.grad(loss, leaf)
            acc = tuple(
                a + eta[:, :, s].view(b, h, 1, *([1] * (gr.dim() - 3))) * gr
                for a, gr in zip(acc, grads, strict=True)
            )
            s_i = scale[s - first]
            current = tuple(p - s_i * a for p, a in zip(start, acc, strict=True))
            outs.append(inner_forward(current, q[:, :, s : s + 1], cfg, norm))
        start = tuple(p.detach() for p in current)
    return torch.cat(outs, dim=2)


@pytest.mark.parametrize("model", ["linear", "mlp"])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.parametrize("minibatch", [1, 4])
def test_ttt_scan_token_scale_matches_published_row_scaling(
    model: InnerModel, use_norm: bool, minibatch: int
) -> None:
    b, h, t, d = 2, 2, 10, 4  # t = 10 is not a multiple of 4: the last chunk has 2 tokens
    g = _gen(1)
    q, k = _l2(torch.randn(b, h, t, d, generator=g)), _l2(torch.randn(b, h, t, d, generator=g))
    v = torch.randn(b, h, t, d, generator=g)
    eta = 0.5 * torch.rand(b, h, t, generator=g)
    cfg = TTTInnerConfig(inner_model=model, inner_norm=use_norm, minibatch=minibatch)
    init = _init(model, h, d, seed=2)
    norm = (1.0 + 0.3 * torch.randn(h, d, generator=g), 0.3 * torch.randn(h, d, generator=g))
    norm_arg = norm if use_norm else None
    # Hand-set offsets, one of them negative enough that the clamp makes s_i = 0.
    offsets = torch.tensor([0.3, -0.2, -0.9, 0.1])[:minibatch]
    scale = torch.clamp_min(1.0 / torch.arange(1, minibatch + 1) + offsets, 0.0)
    with torch.no_grad():
        dual = ttt_scan(q, k, v, eta, init, cfg, norm_arg, token_scale=scale).outputs
        primal = ttt_scan(
            q, k, v, eta, init, cfg, norm_arg, return_states=True, token_scale=scale
        ).outputs
        unscaled = ttt_scan(q, k, v, eta, init, cfg, norm_arg).outputs
    expected = _reference(q, k, v, eta, init, cfg, norm_arg, scale)
    torch.testing.assert_close(dual, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(primal, expected, atol=1e-4, rtol=1e-4)
    # The scaling must change something, or the comparison above proves nothing.
    assert (unscaled - expected).abs().max() > 1e-3


def test_ttt_scan_states_of_short_final_chunk_use_its_own_positions() -> None:
    """Linear, no LN, minibatch 4, T = 6: the final chunk is tokens 4 and 5 (positions 1
    and 2 of their block), so W_4 uses s_1 and W_5 (the carried state) uses s_2."""
    b, h, t, d, mb = 1, 1, 6, 3, 4
    g = _gen(3)
    q, k, v = (torch.randn(b, h, t, d, generator=g) for _ in range(3))
    eta = torch.rand(b, h, t, generator=g)
    cfg = TTTInnerConfig("linear", inner_norm=False, minibatch=mb)
    w0 = torch.randn(h, d, d, generator=g)
    scale = torch.tensor([1.0, 0.5, 0.25, 0.125])
    states = ttt_scan(q, k, v, eta, (w0,), cfg, None, return_states=True, token_scale=scale)
    assert states.weight_states is not None
    w_states = states.weight_states[0][0, 0]
    w_start = w_states[3]  # after the first, full chunk
    step = [eta[0, 0, r] * torch.outer(w_start @ k[0, 0, r] - v[0, 0, r], k[0, 0, r]) for r in (4, 5)]
    torch.testing.assert_close(w_states[4], w_start - 1.0 * step[0])
    torch.testing.assert_close(w_states[5], w_start - 0.5 * (step[0] + step[1]))


def _mixer(options: TTTOptions) -> TTTMixer:
    torch.manual_seed(0)
    mixer = build_mixer(ModelSpec("ttt", d_model=16, n_heads=2, options=options))
    assert isinstance(mixer, TTTMixer)
    return mixer


def test_mixer_token_scale_is_clamped_inverse_position_plus_offset() -> None:
    mixer = _mixer(TTTOptions(minibatch=4, token_eta=True))
    assert mixer.learnable_token_idx is not None
    torch.testing.assert_close(mixer.token_scale(), torch.tensor([1.0, 0.5, 1 / 3, 0.25]))
    with torch.no_grad():
        mixer.learnable_token_idx.copy_(torch.tensor([0.5, -0.7, 0.1, -0.25]))
    torch.testing.assert_close(mixer.token_scale(), torch.tensor([1.5, 0.0, 1 / 3 + 0.1, 0.0]))


@pytest.mark.parametrize("inner_model", ["linear", "mlp"])
def test_mixer_with_token_eta_matches_reference(inner_model: InnerModel) -> None:
    mixer = _mixer(TTTOptions(inner_model=inner_model, minibatch=4, token_eta=True))
    assert mixer.learnable_token_idx is not None
    offsets = torch.tensor([0.2, -0.6, 0.05, 0.1])
    with torch.no_grad():
        mixer.learnable_token_idx.copy_(offsets)
    x = torch.randn(2, 10, 16, generator=_gen(4))
    expected_scale = torch.clamp_min(1.0 / torch.arange(1, 5) + offsets, 0.0)
    with torch.no_grad():
        inputs = mixer.scan_inputs(x)
        got = mixer(x)
    assert inputs.token_scale is not None
    torch.testing.assert_close(inputs.token_scale, expected_scale)
    expected_core = _reference(
        inputs.q, inputs.k, inputs.v, inputs.eta, inputs.init, mixer.cfg, inputs.norm,
        expected_scale,
    )
    with torch.no_grad():
        expected = mixer.output(expected_core.transpose(1, 2).flatten(2), x)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_token_eta_offset_receives_gradient() -> None:
    mixer = _mixer(TTTOptions(minibatch=4, token_eta=True))
    assert mixer.learnable_token_idx is not None
    mixer(torch.randn(2, 10, 16, generator=_gen(5))).pow(2).sum().backward()
    grad = mixer.learnable_token_idx.grad
    assert grad is not None and grad.abs().max() > 0


def test_token_eta_off_is_unchanged() -> None:
    """Default off: no new parameter, and the output is bit-identical to the plain scan."""
    mixer = _mixer(TTTOptions(minibatch=4))
    assert mixer.learnable_token_idx is None
    assert mixer.token_scale() is None
    assert not any("token_idx" in name for name in mixer.state_dict())
    x = torch.randn(2, 10, 16, generator=_gen(6))
    with torch.no_grad():
        q, k, v = mixer.project(x)
        eta = mixer.eta_max * torch.sigmoid(mixer._per_head(mixer.eta_proj, x).squeeze(-1))
        gamma, beta = mixer.ln_weight, mixer.ln_bias
        assert gamma is not None and beta is not None
        expected = ttt_scan(
            q, k.squeeze(3), v.squeeze(3), eta, tuple(mixer.inner_init), mixer.cfg, (gamma, beta)
        ).outputs
        got = mixer.core(x, q, k, v)
    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
