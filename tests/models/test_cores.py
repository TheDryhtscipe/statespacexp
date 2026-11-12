"""Core-recurrence tests: equivalences between scans and representability constructions.

These call the pure scan functions directly with hand-set per-token q, k, v and
transition parameters, as if the mixer projections had produced them from one-hot
tokens. Layout everywhere: [batch, heads, time, dim].
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from sst.models.cores import (
    complex_diag_scan,
    delta_scan,
    diag_scan,
    householder_scan,
    linear_attention,
)
from sst.models.s4d import fft_causal_conv, s4d_initial_response, s4d_kernel, s4d_recurrence
from sst.models.ttt import InnerModel, TTTInnerConfig, inner_forward, inner_grad, ttt_scan


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _l2(x: Tensor) -> Tensor:
    out: Tensor = x / x.norm(dim=-1, keepdim=True)
    return out


def _qkv(b: int, h: int, t: int, dk: int, dv: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    g = _gen(seed)
    q = torch.randn(b, h, t, dk, generator=g)
    k = torch.randn(b, h, t, dk, generator=g)
    v = torch.randn(b, h, t, dv, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# Equivalences
# ---------------------------------------------------------------------------


def test_diag_scan_with_unit_alpha_equals_linear_attention() -> None:
    q, k, v = _qkv(2, 3, 20, 8, 6, seed=0)
    alpha = torch.ones(2, 3, 20, 8)
    scanned = diag_scan(q, k, v, alpha).outputs
    parallel = linear_attention(q, k, v)
    assert scanned.shape == (2, 3, 20, 6)
    assert parallel.abs().max() > 0.1
    torch.testing.assert_close(scanned, parallel, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("beta_max", [1.0, 2.0])
def test_delta_scan_equals_householder_scan_with_one_step(beta_max: float) -> None:
    q, k, v = _qkv(2, 3, 30, 8, 5, seed=1)
    q, k = _l2(q), _l2(k)
    beta = beta_max * torch.rand(2, 3, 30, generator=_gen(2))
    delta = delta_scan(q, k, v, beta, return_states=True)
    product = householder_scan(
        q, k.unsqueeze(-2), v.unsqueeze(-2), beta.unsqueeze(-1), return_states=True
    )
    assert delta.states is not None and product.states is not None
    assert delta.states.abs().max() > 0.1
    torch.testing.assert_close(delta.states, product.states, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(delta.outputs, product.outputs, atol=1e-5, rtol=1e-5)


def test_ttt_linear_online_equals_delta_rule() -> None:
    """TTT-Linear, no LN, no residual, minibatch 1, eta_t = beta_t is the delta rule.

    Inner loss 0.5 * ||W k - v||^2 has gradient (W k - v) k^T, so
    W_t = W_{t-1} - beta (W_{t-1} k - v) k^T = W_{t-1} (I - beta k k^T) + beta v k^T.
    """
    b, h, t, d = 2, 3, 40, 8
    q, k, v = _qkv(b, h, t, d, d, seed=3)
    q, k = _l2(q), _l2(k)
    beta = torch.rand(b, h, t, generator=_gen(4))
    cfg = TTTInnerConfig(inner_model="linear", inner_norm=False, minibatch=1)
    ttt = ttt_scan(q, k, v, beta, (torch.zeros(h, d, d),), cfg, norm=None, return_states=True)
    delta = delta_scan(q, k, v, beta, return_states=True)
    assert ttt.weight_states is not None and delta.states is not None
    assert delta.states.abs().max() > 0.1
    # Same W / S trajectory at every step, and the same outputs.
    torch.testing.assert_close(ttt.weight_states[0], delta.states, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ttt.outputs, delta.outputs, atol=1e-5, rtol=1e-5)
    dual = ttt_scan(q, k, v, beta, (torch.zeros(h, d, d),), cfg, norm=None).outputs
    torch.testing.assert_close(dual, delta.outputs, atol=1e-5, rtol=1e-5)


def _inner_weights(
    model: InnerModel, b: int, h: int, c: int, d: int, seed: int
) -> tuple[Tensor, ...]:
    g = _gen(seed)
    if model == "linear":
        return (torch.randn(b, h, c, d, d, generator=g) * 0.5,)
    hid = 4 * d
    return (
        torch.randn(b, h, c, hid, d, generator=g) * 0.5,
        torch.randn(b, h, c, hid, generator=g) * 0.5,
        torch.randn(b, h, c, d, hid, generator=g) * 0.3,
        torch.randn(b, h, c, d, generator=g) * 0.5,
    )


def _norm_params(h: int, d: int, seed: int) -> tuple[Tensor, Tensor]:
    g = _gen(seed)
    return 1.0 + 0.3 * torch.randn(h, d, generator=g), 0.3 * torch.randn(h, d, generator=g)


@pytest.mark.parametrize("model", ["linear", "mlp"])
@pytest.mark.parametrize("use_norm", [False, True])
def test_ttt_inner_grad_matches_autograd(model: InnerModel, use_norm: bool) -> None:
    b, h, c, d = 2, 3, 5, 6
    cfg = TTTInnerConfig(inner_model=model, inner_norm=use_norm, minibatch=1)
    weights = tuple(w.requires_grad_(True) for w in _inner_weights(model, b, h, c, d, seed=5))
    norm = _norm_params(h, d, seed=6) if use_norm else None
    g = _gen(7)
    k = torch.randn(b, h, c, d, generator=g)
    v = torch.randn(b, h, c, d, generator=g)
    # Each token has its own copy of the weights, so autograd of the summed loss
    # gives the per-token gradients.
    loss = 0.5 * (inner_forward(weights, k, cfg, norm) - v).pow(2).sum()
    expected = torch.autograd.grad(loss, weights)
    got = inner_grad(weights, k, v, cfg, norm)
    assert len(got) == len(expected)
    for e, a in zip(expected, got, strict=True):
        assert e.abs().max() > 1e-3
        torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-4)


def _ttt_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    eta: Tensor,
    init: tuple[Tensor, ...],
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
) -> Tensor:
    """Sun et al. (2024) mini-batch TTT, written token by token with autograd gradients.

    Within a mini-batch every gradient is taken at W_0 (the weights at the start of the
    mini-batch); token s uses W_s = W_0 - sum_{r <= s} eta_r grad l(W_0; k_r, v_r).
    """
    b, h, t, _ = q.shape
    w0 = tuple(p.unsqueeze(0).unsqueeze(2).expand(b, h, 1, *p.shape[1:]).clone() for p in init)
    outs: list[Tensor] = []
    for start in range(0, t, cfg.minibatch):
        acc = tuple(torch.zeros_like(p) for p in w0)
        current = w0
        for s in range(start, min(start + cfg.minibatch, t)):
            leaf = tuple(p.detach().requires_grad_(True) for p in w0)
            ks, vs = k[:, :, s : s + 1], v[:, :, s : s + 1]
            loss = 0.5 * (inner_forward(leaf, ks, cfg, norm) - vs).pow(2).sum()
            grads = torch.autograd.grad(loss, leaf)
            acc = tuple(
                a + eta[:, :, s].view(b, h, 1, *([1] * (gr.dim() - 3))) * gr
                for a, gr in zip(acc, grads, strict=True)
            )
            current = tuple(p - a for p, a in zip(w0, acc, strict=True))
            outs.append(inner_forward(current, q[:, :, s : s + 1], cfg, norm))
        w0 = tuple(p.detach() for p in current)
    return torch.cat(outs, dim=2)


@pytest.mark.parametrize("model", ["linear", "mlp"])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.parametrize("minibatch", [1, 4])
def test_ttt_scan_matches_minibatch_reference(
    model: InnerModel, use_norm: bool, minibatch: int
) -> None:
    b, h, t, d = 2, 2, 10, 4
    q, k, v = _qkv(b, h, t, d, d, seed=8)
    q, k = _l2(q), _l2(k)
    eta = 0.3 * torch.rand(b, h, t, generator=_gen(9))  # small enough that inner SGD stays bounded
    cfg = TTTInnerConfig(inner_model=model, inner_norm=use_norm, minibatch=minibatch)
    init = tuple(w[0, :, 0] for w in _inner_weights(model, 1, h, 1, d, seed=10))
    norm = _norm_params(h, d, seed=11) if use_norm else None
    with torch.no_grad():
        dual = ttt_scan(q, k, v, eta, init, cfg, norm).outputs
        primal = ttt_scan(q, k, v, eta, init, cfg, norm, return_states=True).outputs
    expected = _ttt_reference(q, k, v, eta, init, cfg, norm)
    assert expected.abs().max() > 1e-2
    torch.testing.assert_close(dual, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(primal, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model", ["linear", "mlp"])
def test_ttt_scan_is_differentiable_for_outer_training(model: InnerModel) -> None:
    """gradcheck in float64: the analytic inner gradient is itself differentiable."""
    b, h, t, d = 1, 1, 4, 3
    cfg = TTTInnerConfig(inner_model=model, inner_norm=True, minibatch=2)
    q, k, v = (x.double() for x in _qkv(b, h, t, d, d, seed=12))
    eta = torch.rand(b, h, t, generator=_gen(13)).double()
    init = tuple(w[0, :, 0].double() for w in _inner_weights(model, 1, h, 1, d, seed=14))
    gamma, beta = (p.double() for p in _norm_params(h, d, seed=15))
    inputs = (q, k, v, eta, gamma, beta, *init)
    leaves = tuple(x.clone().requires_grad_(True) for x in inputs)

    def fn(*args: Tensor) -> Tensor:
        qq, kk, vv, ee, gg, bb = args[:6]
        return ttt_scan(qq, kk, vv, ee, tuple(args[6:]), cfg, (gg, bb)).outputs

    out = fn(*leaves)
    assert out.abs().max() > 1e-3
    # The eta and inner-weight gradients must be non-zero: the outer loss sees the inner update.
    grads = torch.autograd.grad(out.sum(), leaves)
    assert grads[3].abs().max() > 1e-6 and grads[6].abs().max() > 1e-6
    assert torch.autograd.gradcheck(fn, leaves, eps=1e-6, atol=1e-5)


def test_s4d_fft_convolution_equals_sequential_recurrence() -> None:
    channels, modes, t = 4, 8, 300
    g = _gen(16)
    # Slow decay (|lambda| close to 1) so that circular wrap-around would show.
    dt = torch.full((channels, 1), 1e-3)
    a = torch.complex(-0.5 * torch.ones(channels, modes), math.pi * torch.arange(modes).float())
    lam = torch.exp(dt * a)
    b_bar = torch.complex(torch.randn(channels, modes, generator=g), torch.randn(channels, modes, generator=g))
    c = torch.complex(torch.randn(channels, modes, generator=g), torch.randn(channels, modes, generator=g))
    u = torch.randn(2, t, channels, generator=g)
    kernel = s4d_kernel(lam, b_bar, c, t)
    assert kernel.shape == (channels, t)
    y_fft = fft_causal_conv(u, kernel)
    y_rec = s4d_recurrence(u, lam, b_bar, c)
    assert y_rec.abs().max() > 1.0
    torch.testing.assert_close(y_fft, y_rec, atol=1e-3, rtol=1e-4)
    # With an initial state x0: convolution + 2 Re(c lam^{t+1} x0) equals the recurrence.
    x0 = torch.complex(torch.randn(channels, modes, generator=g), torch.randn(channels, modes, generator=g))
    y_rec0 = s4d_recurrence(u, lam, b_bar, c, initial_state=x0)
    assert (y_rec0 - y_rec).abs().max() > 1e-2
    torch.testing.assert_close(y_fft + s4d_initial_response(lam, c, x0, t), y_rec0, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# Representability constructions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shrink", [0.0, 1e-3])
def test_diag_neg_tracks_parity_to_length_512(shrink: float) -> None:
    """alpha(x) = 1 - 2x flips the state's sign on every 1: S_t = (-1)^{parity} S_0.

    shrink > 0 keeps alpha strictly inside the open interval (-1, 1) that the
    diag_neg parametrisation can reach.
    """
    b, t = 16, 512
    bits = torch.randint(0, 2, (b, t), generator=_gen(17))
    alpha = ((1.0 - 2.0 * bits.float()) * (1.0 - shrink)).view(b, 1, t, 1)
    k = torch.zeros(b, 1, t, 1)
    v = torch.zeros(b, 1, t, 2)
    q = torch.ones(b, 1, t, 1)
    s0 = torch.tensor([[1.0], [-1.0]]).expand(b, 1, 2, 1)
    out = diag_scan(q, k, v, alpha, initial_state=s0).outputs[:, 0]  # [b, t, 2]
    parity = bits.cumsum(dim=1) % 2
    assert torch.equal(out.argmax(dim=-1), parity)


def _swap_columns(s: Tensor, i: int, j: int) -> Tensor:
    out = s.clone()
    out[..., [i, j]] = s[..., [j, i]]
    return out


def _transposition_key(i: int, j: int, n: int) -> Tensor:
    key = torch.zeros(n)
    key[i] = 1.0
    key[j] = -1.0
    return key / math.sqrt(2.0)


def test_beta_two_householder_is_a_transposition() -> None:
    n = 5
    s0 = torch.randn(1, 1, n, n, generator=_gen(18))
    k = _transposition_key(1, 3, n).view(1, 1, 1, n)
    v = torch.zeros(1, 1, 1, n)
    q = torch.zeros(1, 1, 1, n)
    beta = torch.full((1, 1, 1), 2.0)
    expected = _swap_columns(s0, 1, 3)
    by_delta = delta_scan(q, k, v, beta, initial_state=s0).final_state
    by_product = householder_scan(
        q, k.unsqueeze(-2), v.unsqueeze(-2), beta.unsqueeze(-1), initial_state=s0
    ).final_state
    torch.testing.assert_close(by_delta, expected)
    torch.testing.assert_close(by_product, expected)


def _transpositions_for(perm: list[int]) -> list[tuple[int, int]]:
    """Transpositions (in application order) whose matrices multiply to P(perm).

    P(perm) has P[perm[i], i] = 1. Right-multiplying by a transposition swaps two
    columns, so sorting the columns of P back to I gives P T_1 ... T_m = I, hence
    P = T_m ... T_1.
    """
    n = len(perm)
    cols = list(perm)  # cols[i] = row index of the 1 in column i
    swaps: list[tuple[int, int]] = []
    for i in range(n):
        j = cols.index(i)
        if j != i:
            cols[i], cols[j] = cols[j], cols[i]
            swaps.append((i, j))
    return list(reversed(swaps))


def _perm_matrix(perm: list[int]) -> Tensor:
    n = len(perm)
    m = torch.zeros(n, n)
    for i, p in enumerate(perm):
        m[p, i] = 1.0
    return m


@pytest.mark.parametrize("shrink", [0.0, 1e-4])
def test_deltaproduct_four_steps_tracks_s5_to_length_512(shrink: float) -> None:
    n, n_h, b, t = 5, 4, 4, 512
    g = _gen(19)
    perms = [[torch.randperm(n, generator=g).tolist() for _ in range(t)] for _ in range(b)]
    k = torch.zeros(b, 1, t, n_h, n)
    k[..., 0] = 1.0  # padding steps: any unit key with beta = 0
    beta = torch.zeros(b, 1, t, n_h)
    for bi in range(b):
        for ti in range(t):
            swaps = _transpositions_for(perms[bi][ti])
            assert len(swaps) <= n_h
            product = torch.eye(n)
            for step, (i, j) in enumerate(swaps):
                k[bi, 0, ti, step] = _transposition_key(i, j, n)
                beta[bi, 0, ti, step] = 2.0 * (1.0 - shrink)
                product = product @ (torch.eye(n) - 2.0 * torch.outer(k[bi, 0, ti, step], k[bi, 0, ti, step]))
            torch.testing.assert_close(product, _perm_matrix(perms[bi][ti]), atol=1e-6, rtol=0.0)
    v = torch.zeros(b, 1, t, n_h, n)
    q = torch.zeros(b, 1, t, n)
    s0 = torch.eye(n).expand(b, 1, n, n)
    states = householder_scan(q, k, v, beta, initial_state=s0, return_states=True).states
    assert states is not None
    # Read every column with q = e_c (all five queries at once): S_t e_c.
    reads = torch.einsum("bhtij,jc->bhtic", states, torch.eye(n))[:, 0]  # [b, t, row, c]
    decoded = reads.argmax(dim=2)  # [b, t, c]: row of the 1 in column c
    # Integer reference: M_t = M_{t-1} P(pi_t), so column i of M_t is column pi_t(i) of M_{t-1}.
    expected = torch.zeros(b, t, n, dtype=torch.long)
    for bi in range(b):
        cols = list(range(n))
        for ti in range(t):
            cols = [cols[p] for p in perms[bi][ti]]
            expected[bi, ti] = torch.tensor(cols)
    assert torch.equal(decoded, expected)


def test_diag_complex_tracks_z3_to_length_512() -> None:
    """theta = 2 pi / 3 per increment rotates a unit state through the three cube roots of 1."""
    b, t = 16, 512
    x = torch.randint(0, 3, (b, t), generator=_gen(20))
    angle = (2.0 * math.pi / 3.0 * x.float()).view(b, 1, t, 1)
    radius = torch.ones(b, 1, t, 1)
    k = torch.zeros(b, 1, t, 2)
    v = torch.zeros(b, 1, t, 1)
    q = torch.zeros(b, 1, t, 2)
    s0 = torch.tensor([[1.0, 0.0]]).expand(b, 1, 1, 2)
    states = complex_diag_scan(q, k, v, radius, angle, initial_state=s0, return_states=True).states
    assert states is not None
    phases = 2.0 * math.pi / 3.0 * torch.arange(3).float()
    queries = torch.stack([phases.cos(), phases.sin()])  # [2, 3]: q_c for class c
    logits = torch.einsum("btvk,kc->btc", states[:, 0], queries)  # S_t q_c (dv = 1)
    assert torch.equal(logits.argmax(dim=-1), x.cumsum(dim=1) % 3)
