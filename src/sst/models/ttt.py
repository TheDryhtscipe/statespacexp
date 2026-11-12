"""Test-time training (TTT) layer core (Sun et al. 2024, "Learning to (Learn at Test Time):
RNNs with Expressive Hidden States").

The hidden state is the weight set W of an inner model f_W. Per token the inner loss is
l(W; k, v) = 1/2 ||f_W(k) - v||^2 and the output is f_{W_t}(q_t). The paper's loss has
no 1/2, so at the same learning rate eta the step here is half the paper's.

Inner model (per head, dk = dv = d):
  - "linear": g(x) = W x                      (weights: (W,))
  - "mlp":    g(x) = W2 gelu(W1 x + b1) + b2  (weights: (W1, b1, W2, b2), hidden 4d)
  - inner_norm=True:  f(x) = x + LN(g(x)) with a learnable per-head LN (Sun et al.);
    inner_norm=False: f(x) = g(x), no residual.

Mini-batch TTT (Sun et al., Sec. 2.7): with mini-batch size b and t' the last token of
the previous mini-batch, every token s in the current mini-batch takes its gradient at
W_{t'}, and

    W_t = W_{t'} - sum_{s = t'+1}^{t} eta_s grad l(W_{t'}; k_s, v_s),   z_t = f_{W_t}(q_t).

So token t reads the weights after the cumulative in-mini-batch update up to and
including itself (this is what the paper's dual form computes); the next mini-batch
starts from W at its last token. b = 1 is fully online gradient descent.

Gradients are analytic (no nested autograd), built from differentiable ops, so the outer
training loop can backpropagate through the inner updates. Tests compare them with
torch.autograd.

Layout: q, k, v [B, H, T, d]; per-token weights [B, H, c, *shape] where c is the number
of tokens (1 broadcasts over tokens).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import Tensor

InnerModel = Literal["linear", "mlp"]

LN_EPS = 1e-5


@dataclass(frozen=True)
class TTTInnerConfig:
    inner_model: InnerModel
    inner_norm: bool
    minibatch: int


class TTTScanResult(NamedTuple):
    """outputs [B, H, T, dv]; weight_states: per weight tensor, its value after each token
    ([B, H, T, *shape]), when requested."""

    outputs: Tensor
    weight_states: tuple[Tensor, ...] | None


class TTTScanInputs(NamedTuple):
    """The arguments a TTT mixer passes to ttt_scan: q, k, v [B, H, T, d], eta [B, H, T],
    init per-head W_0 tensors [H, *shape], inner LN (gamma, beta) [H, d] or None, and
    token_scale [minibatch] or None."""

    q: Tensor
    k: Tensor
    v: Tensor
    eta: Tensor
    init: tuple[Tensor, ...]
    norm: tuple[Tensor, Tensor] | None
    token_scale: Tensor | None


def _matvec(w: Tensor, x: Tensor) -> Tensor:
    """w [..., m, n] @ x [..., n] -> [..., m], with broadcasting over leading dims.

    When w is shared across the token dim (w [B, H, 1, m, n], x [B, H, c, n]) this is
    x @ w^T, which avoids the broadcasting matmul copying w once per token.
    """
    if w.dim() == x.dim() + 1 and w.shape[-3] == 1 and x.dim() >= 2:
        return x @ w.squeeze(-3).transpose(-1, -2)
    return (w @ x.unsqueeze(-1)).squeeze(-1)


def _gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _gelu_grad(x: Tensor) -> Tensor:
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf


def _norm_view(p: Tensor, ndim: int) -> Tensor:
    """Per-head LN parameter [H, d] -> [H, 1, ..., d] to broadcast against [B, H, c, d]."""
    return p.view(p.shape[0], *([1] * (ndim - 3)), p.shape[1])


class _Forward(NamedTuple):
    out: Tensor  # f(x)
    y: Tensor  # g(x)
    hidden_pre: Tensor | None  # W1 x + b1 (mlp)
    hidden: Tensor | None  # gelu(W1 x + b1) (mlp)
    x_hat: Tensor | None  # normalised g(x) (inner_norm)
    inv_std: Tensor | None  # 1 / sqrt(var + eps) (inner_norm)


def _forward(
    weights: tuple[Tensor, ...],
    x: Tensor,
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
) -> _Forward:
    hidden_pre: Tensor | None = None
    hidden: Tensor | None = None
    if cfg.inner_model == "linear":
        (w,) = weights
        y = _matvec(w, x)
    else:
        w1, b1, w2, b2 = weights
        hidden_pre = _matvec(w1, x) + b1
        hidden = _gelu(hidden_pre)
        y = _matvec(w2, hidden) + b2
    out, x_hat, inv_std = _finish(y, x, cfg, norm)
    return _Forward(out, y, hidden_pre, hidden, x_hat, inv_std)


def _finish(
    y: Tensor, x: Tensor, cfg: TTTInnerConfig, norm: tuple[Tensor, Tensor] | None
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """f = g(x), or x + LN(g(x)) with inner_norm. Returns (f, x_hat, 1 / std)."""
    if not cfg.inner_norm:
        return y, None, None
    if norm is None:
        raise ValueError("inner_norm=True needs LN parameters")
    gamma, beta = (_norm_view(p, y.dim()) for p in norm)
    centred = y - y.mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(centred.pow(2).mean(dim=-1, keepdim=True) + LN_EPS)
    x_hat = centred * inv_std
    return x + gamma * x_hat + beta, x_hat, inv_std


def inner_forward(
    weights: tuple[Tensor, ...],
    x: Tensor,
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
) -> Tensor:
    """f_W(x) for x [B, H, c, d] and weights [B, H, c or 1, *shape]."""
    return _forward(weights, x, cfg, norm).out


def inner_grad(
    weights: tuple[Tensor, ...],
    k: Tensor,
    v: Tensor,
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
) -> tuple[Tensor, ...]:
    """Per-token gradient of 1/2 ||f_W(k) - v||^2 with respect to each weight tensor.

    Returns tensors of shape [B, H, c, *shape] (one gradient per token).
    """
    dy, d_hidden, hidden = _grad_factors(weights, k, v, cfg, norm)
    if cfg.inner_model == "linear":
        return (_outer(dy, k),)
    if d_hidden is None or hidden is None:
        raise ValueError("mlp forward cache missing")
    return (_outer(d_hidden, k), d_hidden, _outer(dy, hidden), dy)


def _outer(a: Tensor, b: Tensor) -> Tensor:
    return a.unsqueeze(-1) * b.unsqueeze(-2)


def _grad_factors(
    weights: tuple[Tensor, ...],
    k: Tensor,
    v: Tensor,
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """The rank-one factors of the per-token gradients: (dl/dy, dl/d(hidden_pre), hidden).

    linear: grad W = dy k^T.  mlp: grad W1 = dh k^T, grad b1 = dh, grad W2 = dy hidden^T,
    grad b2 = dy.
    """
    fwd = _forward(weights, k, cfg, norm)
    err = fwd.out - v  # dl/df
    if cfg.inner_norm:
        if norm is None or fwd.x_hat is None or fwd.inv_std is None:
            raise ValueError("inner_norm=True needs LN parameters")
        gamma = _norm_view(norm[0], err.dim())
        g_hat = gamma * err
        dy = fwd.inv_std * (
            g_hat
            - g_hat.mean(dim=-1, keepdim=True)
            - fwd.x_hat * (g_hat * fwd.x_hat).mean(dim=-1, keepdim=True)
        )
    else:
        dy = err
    if cfg.inner_model == "linear":
        return dy, None, None
    if fwd.hidden is None or fwd.hidden_pre is None:
        raise ValueError("mlp forward cache missing")
    _, _, w2, _ = weights
    d_hidden = _matvec(w2.transpose(-1, -2), dy) * _gelu_grad(fwd.hidden_pre)
    return dy, d_hidden, fwd.hidden


def _dual_chunk(
    weights: tuple[Tensor, ...],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    eta: Tensor,
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
    scale: Tensor | None = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """One mini-batch in the dual form (Sun et al., Sec. 2.5): the same outputs as the
    primal form without materialising per-token weights.

    Every gradient is rank one, so with W_s = W_0 - sum_{r <= s} eta_r a_r b_r^T,
    W_s x = W_0 x - sum_{r <= s} eta_r (b_r . x) a_r: a causally masked, attention-like
    sum. weights: [B, H, 1, *shape]; q, k, v: [B, H, c, d]; eta: [B, H, c].

    scale [c] (token_eta): the readout at position s uses s_s * sum_{r <= s}, so the
    mask entry is s_s eta_r, and the carried weights use the last position's s_c.
    """
    dy, d_hidden, hidden_k = _grad_factors(weights, k, v, cfg, norm)
    c = q.shape[2]
    causal = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()
    mask = eta.unsqueeze(-2) * causal  # [B, H, c(s), c(r)] = eta_r [r <= s]
    if scale is not None:
        mask = scale.unsqueeze(-1) * mask
        eta = scale[-1] * eta  # the carried weights are the last position's
    if cfg.inner_model == "linear":
        (w,) = weights
        y = _matvec(w, q) - (mask * (q @ k.transpose(-1, -2))) @ dy
        new_w = w - torch.einsum("bhr,bhrv,bhrk->bhvk", eta, dy, k).unsqueeze(2)
        return _finish(y, q, cfg, norm)[0], (new_w,)
    if d_hidden is None or hidden_k is None:
        raise ValueError("mlp forward cache missing")
    w1, b1, w2, b2 = weights
    # The "+ 1" terms are the bias gradients (their input is the constant 1).
    pre = _matvec(w1, q) + b1 - (mask * (q @ k.transpose(-1, -2) + 1.0)) @ d_hidden
    hidden_q = _gelu(pre)
    y = _matvec(w2, hidden_q) + b2 - (mask * (hidden_q @ hidden_k.transpose(-1, -2) + 1.0)) @ dy
    new = (
        w1 - torch.einsum("bhr,bhrm,bhrk->bhmk", eta, d_hidden, k).unsqueeze(2),
        b1 - torch.einsum("bhr,bhrm->bhm", eta, d_hidden).unsqueeze(2),
        w2 - torch.einsum("bhr,bhrv,bhrm->bhvm", eta, dy, hidden_k).unsqueeze(2),
        b2 - torch.einsum("bhr,bhrv->bhv", eta, dy).unsqueeze(2),
    )
    return _finish(y, q, cfg, norm)[0], new


def ttt_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    eta: Tensor,
    init_weights: tuple[Tensor, ...],
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
    return_states: bool = False,
    token_scale: Tensor | None = None,
) -> TTTScanResult:
    """Run mini-batch TTT over the sequence.

    eta: [B, H, T] per-token inner learning rate. init_weights: per-head initial weights
    [H, *shape], shared by every sequence in the batch. Without return_states the dual
    form runs (same outputs, far less memory), except for the linear model with
    minibatch 1, where the primal form is cheaper. The primal form materialises
    per-token weights and can return them.

    token_scale [minibatch] (TTTOptions.token_eta, published ttt-lm-pytorch scaling):
    position i (1-based) of each mini-batch reads
    W_i = W_start - token_scale[i-1] * sum_{r <= i} eta_r grad_r. Positions count from
    the start of each torch.split chunk, so a short final chunk of c tokens uses
    token_scale[:c] and carries W_c forward. None means no scaling (all ones).
    """
    if cfg.minibatch < 1:
        raise ValueError("minibatch must be >= 1")
    if token_scale is not None and token_scale.shape != (cfg.minibatch,):
        raise ValueError("token_scale must have shape [minibatch]")
    b, h, _, _ = q.shape
    weights = tuple(p.unsqueeze(0).unsqueeze(2).expand(b, h, 1, *p.shape[1:]) for p in init_weights)
    outputs: list[Tensor] = []
    trajectory: list[tuple[Tensor, ...]] = []
    # split (not slicing in the loop) so the backward pass concatenates once.
    chunks = zip(
        torch.split(q, cfg.minibatch, dim=2),
        torch.split(k, cfg.minibatch, dim=2),
        torch.split(v, cfg.minibatch, dim=2),
        torch.split(eta, cfg.minibatch, dim=2),
        strict=True,
    )
    for q_c, k_c, v_c, eta_c in chunks:
        scale_c = None if token_scale is None else token_scale[: q_c.shape[2]]
        if not return_states and (cfg.minibatch > 1 or cfg.inner_model == "mlp"):
            out_c, weights = _dual_chunk(weights, q_c, k_c, v_c, eta_c, cfg, norm, scale_c)
            outputs.append(out_c)
            continue
        grads = inner_grad(weights, k_c, v_c, cfg, norm)
        if scale_c is None:
            per_token = tuple(
                w - torch.cumsum(eta_c.view(*eta_c.shape, *([1] * (g.dim() - 3))) * g, dim=2)
                for w, g in zip(weights, grads, strict=True)
            )
        else:
            per_token = tuple(
                w
                - scale_c.view(-1, *([1] * (g.dim() - 3)))
                * torch.cumsum(eta_c.view(*eta_c.shape, *([1] * (g.dim() - 3))) * g, dim=2)
                for w, g in zip(weights, grads, strict=True)
            )
        outputs.append(inner_forward(per_token, q_c, cfg, norm))
        trajectory.append(per_token)
        weights = tuple(w[:, :, -1:] for w in per_token)
    states = (
        tuple(torch.cat(parts, dim=2) for parts in zip(*trajectory, strict=True))
        if return_states
        else None
    )
    return TTTScanResult(torch.cat(outputs, dim=2), states)
