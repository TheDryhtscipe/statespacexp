"""Overshoot ratio of trained TTT layers: inner step size times inner-loss curvature.

Per token t of a TTT mixer (see sst.models.ttt), the inner loss is
l_t(W) = 1/2 ||f_W(k_t) - v_t||^2 and its gradient is taken at W_start(t): W_{t-1} for
minibatch 1, the weights at the start of t's mini-batch otherwise. This module reports

    curvature_t = lambda_max(Hess l_t(W_start(t)))          (all inner weights, flattened)
    ratio_t     = eta_t * curvature_t
    block_ratio = lambda_max(sum_{s in block} eta_s Hess l_s(W_block_start))

where lambda_max is the largest algebraic (most positive) eigenvalue; the Hessian can be
indefinite (MLP inner model, inner LayerNorm). eta_t is the effective rate with which the
token's gradient enters the weights carried to the next block: the mixer's eta_t times,
with token_eta, the scale s_c of the block's last position (s_1 for minibatch 1).

The carried update of one block is W -> W - grad L(W) with L = sum_s eta_s l_s, which is
the gradient of the convex function 1/2 ||W||^2 - L(W) whenever block_ratio <= 1.

Eigenvalues come from batched Lanczos with full reorthogonalisation on Hessian-vector
products (autograd double-backward of inner_forward), one independent problem per
(sequence, head, token) or (sequence, head, block). Lanczos Ritz values approach the
largest algebraic eigenvalue from below whatever the sign of the rest of the spectrum,
so no spectral shift is needed. The residual ||H y - theta y|| of the top Ritz pair is
reported so callers can check convergence when iterations < the weight dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from sst.models.mixers import TTTMixer
from sst.models.model import SequenceModel
from sst.models.ttt import TTTInnerConfig, inner_forward, ttt_scan


class Eigen(NamedTuple):
    """Largest algebraic eigenvalue per problem [N] and its Ritz residual norm [N]."""

    value: Tensor
    residual: Tensor


class OvershootResult(NamedTuple):
    """For one TTT layer, per sequence and head ([B, H, T], or [B, H, n_blocks]).

    eta: effective inner rate (eta_t times the block-end token_eta scale, if any).
    curvature: lambda_max of Hess l_t at W_start(t). ratio: eta * curvature.
    block_ratio: lambda_max(sum_s eta_s Hess l_s(W_block_start)) per mini-batch; for
    minibatch 1 a block is one token and this equals ratio.
    curvature_residual, block_residual: Lanczos residual norms of the two eigenvalues
    (block_residual is unscaled by eta, like the Hessian it belongs to).
    """

    eta: Tensor
    curvature: Tensor
    ratio: Tensor
    block_ratio: Tensor
    curvature_residual: Tensor
    block_residual: Tensor


def largest_eigenvalue(
    hvp: Callable[[Tensor], Tensor],
    n_items: int,
    dim: int,
    iterations: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Eigen:
    """Largest algebraic eigenvalue of n_items symmetric dim x dim operators.

    hvp maps vectors [n_items, dim] to the products [n_items, dim] (each row with its own
    operator). Lanczos with full reorthogonalisation (applied twice); if a Krylov space
    becomes invariant (beta ~ 0) it restarts from a random vector orthogonal to the basis,
    so low-rank operators are handled. iterations is capped at dim; at dim the result is
    exact up to rounding.
    """
    steps = min(iterations, dim)
    gen = torch.Generator(device=device if device is not None else "cpu").manual_seed(seed)
    tol = torch.finfo(dtype).eps ** 0.5

    def random_unit(basis: Tensor | None) -> Tensor:
        r = torch.randn(n_items, dim, generator=gen, dtype=dtype, device=device)
        if basis is not None:
            r = _orthogonalise(r, basis)
        unit: Tensor = r / r.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        return unit

    basis = torch.zeros(steps, n_items, dim, dtype=dtype, device=device)
    alphas = torch.zeros(n_items, steps, dtype=dtype, device=device)
    betas = torch.zeros(n_items, max(steps - 1, 0), dtype=dtype, device=device)
    scale = torch.zeros(n_items, dtype=dtype, device=device)  # running estimate of ||H||
    q = random_unit(None)
    last_beta = torch.zeros(n_items, dtype=dtype, device=device)
    for j in range(steps):
        basis[j] = q
        w = hvp(q).to(dtype)
        alpha = (w * q).sum(-1)
        alphas[:, j] = alpha
        w = _orthogonalise(w, basis[: j + 1])
        beta = w.norm(dim=-1)
        scale = torch.maximum(scale, torch.maximum(alpha.abs(), beta))
        last_beta = beta
        if j == steps - 1:
            break
        invariant = beta <= tol * scale
        restart = random_unit(basis[: j + 1])
        q = torch.where(
            invariant.unsqueeze(-1), restart, w / beta.clamp_min(torch.finfo(dtype).tiny).unsqueeze(-1)
        )
        betas[:, j] = torch.where(invariant, torch.zeros_like(beta), beta)
    tri = torch.diag_embed(alphas) + torch.diag_embed(betas, 1) + torch.diag_embed(betas, -1)
    evals, evecs = torch.linalg.eigh(tri)
    residual = last_beta * evecs[:, -1, -1].abs()
    return Eigen(evals[:, -1], residual)


def _orthogonalise(w: Tensor, basis: Tensor) -> Tensor:
    """Remove the components of w [N, n] along basis [m, N, n] (twice, for stability)."""
    for _ in range(2):
        coeffs = torch.einsum("mNn,Nn->mN", basis, w)
        w = w - torch.einsum("mN,mNn->Nn", coeffs, basis)
    return w


def mixer_overshoot(
    mixer: TTTMixer,
    x: Tensor,
    iterations: int = 64,
    batch_chunk: int = 4,
    token_chunk: int = 16,
    seed: int = 0,
) -> OvershootResult:
    """Overshoot ratios of one TTT mixer on its input x [B, T, d_model].

    Sequences are processed batch_chunk at a time and Hessian problems token_chunk token
    positions (or token_chunk // minibatch blocks) at a time, each with its own autograd
    graph, so memory is bounded by the chunk sizes and the Lanczos basis
    (iterations x chunk x weight dimension). No gradients reach the mixer's parameters.
    The defaults keep an MLP inner model with head dim 32 (8352 inner weights per head)
    under 1 GiB at B = 16, T = 512, 4 heads. Lanczos runs in the mixer's dtype.
    """
    with torch.no_grad():
        inputs = mixer.scan_inputs(x)
    q, k, v, eta = (t.detach() for t in (inputs.q, inputs.k, inputs.v, inputs.eta))
    init = tuple(p.detach() for p in inputs.init)
    norm = None if inputs.norm is None else (inputs.norm[0].detach(), inputs.norm[1].detach())
    token_scale = None if inputs.token_scale is None else inputs.token_scale.detach()
    cfg = mixer.cfg
    b, h, t, _ = q.shape
    mb = cfg.minibatch
    n_blocks = -(-t // mb)
    block_of = torch.arange(t, device=x.device) // mb
    chunk_len = torch.clamp_max(t - block_of * mb, mb)  # length of each token's block
    carried = torch.ones(t, dtype=eta.dtype, device=x.device)
    if token_scale is not None:
        carried = token_scale[chunk_len - 1]
    eta_eff = eta * carried
    dim = sum(p[0].numel() for p in init)
    steps = min(iterations, dim)

    curvature = torch.zeros(b, h, t, dtype=eta.dtype, device=x.device)
    curvature_res = torch.zeros_like(curvature)
    block_ratio = torch.zeros(b, h, n_blocks, dtype=eta.dtype, device=x.device)
    block_res = torch.zeros_like(block_ratio)
    for b0 in range(0, b, batch_chunk):
        bs = slice(b0, min(b0 + batch_chunk, b))
        with torch.no_grad():
            states = ttt_scan(
                q[bs], k[bs], v[bs], eta[bs], init, cfg, norm, True, token_scale
            ).weight_states
        if states is None:
            raise RuntimeError("ttt_scan returned no weight states")
        nb = q[bs].shape[0]
        # Weights at the start of each block: W_0, then the state after each block's end.
        starts = tuple(
            torch.cat(
                (p.unsqueeze(0).unsqueeze(2).expand(nb, h, 1, *p.shape[1:]), s[:, :, mb - 1 :: mb]),
                dim=2,
            )[:, :, :n_blocks]
            for p, s in zip(init, states, strict=True)
        )
        del states
        for t0 in range(0, t, token_chunk):
            ts = slice(t0, min(t0 + token_chunk, t))
            weights = tuple(w[:, :, block_of[ts]] for w in starts)
            ones = torch.ones(nb, h, ts.stop - t0, dtype=eta.dtype, device=x.device)
            eig = _hessian_eigen(
                weights, None, k[bs, :, ts], v[bs, :, ts], ones, cfg, norm, steps, seed + t0
            )
            curvature[bs, :, ts] = eig.value.view(nb, h, -1)
            curvature_res[bs, :, ts] = eig.residual.view(nb, h, -1)
        if mb == 1:
            continue
        per = max(1, token_chunk // mb)
        for j0 in range(0, n_blocks, per):
            js = slice(j0, min(j0 + per, n_blocks))
            ts = slice(j0 * mb, min(js.stop * mb, t))
            weights = tuple(w[:, :, js] for w in starts)
            eig = _hessian_eigen(
                weights,
                block_of[ts] - j0,
                k[bs, :, ts],
                v[bs, :, ts],
                eta_eff[bs, :, ts],
                cfg,
                norm,
                steps,
                seed + j0,
            )
            block_ratio[bs, :, js] = eig.value.view(nb, h, -1)
            block_res[bs, :, js] = eig.residual.view(nb, h, -1)
    ratio = eta_eff * curvature
    if mb == 1:
        block_ratio, block_res = ratio.clone(), curvature_res.clone()
    return OvershootResult(eta_eff, curvature, ratio, block_ratio, curvature_res, block_res)


def _hessian_eigen(
    weights: tuple[Tensor, ...],
    token_to_item: Tensor | None,
    k: Tensor,
    v: Tensor,
    coeff: Tensor,
    cfg: TTTInnerConfig,
    norm: tuple[Tensor, Tensor] | None,
    steps: int,
    seed: int,
) -> Eigen:
    """lambda_max of the Hessian of sum_t coeff_t l_t with respect to each item's weights.

    weights [nb, H, n_items, *shape] are the independent problems; token t (of k, v
    [nb, H, c, d]) reads item token_to_item[t] (None: token t is item t).
    """
    nb, h, n_items = weights[0].shape[:3]
    shapes = [w.shape[3:] for w in weights]
    sizes = [w[0, 0, 0].numel() for w in weights]
    dim = sum(sizes)
    with torch.enable_grad():
        leaves = tuple(w.detach().clone().requires_grad_(True) for w in weights)
        per_token = leaves if token_to_item is None else tuple(w[:, :, token_to_item] for w in leaves)
        err = inner_forward(per_token, k, cfg, norm) - v
        loss = 0.5 * (coeff * err.pow(2).sum(-1)).sum()
        grads = torch.autograd.grad(loss, leaves, create_graph=True)

        def hvp(vec: Tensor) -> Tensor:
            parts = torch.split(vec.to(weights[0].dtype).view(nb, h, n_items, dim), sizes, dim=-1)
            directions = tuple(
                p.reshape(nb, h, n_items, *sh) for p, sh in zip(parts, shapes, strict=True)
            )
            prods = torch.autograd.grad(grads, leaves, directions, retain_graph=True)
            return torch.cat([p.reshape(nb * h * n_items, -1) for p in prods], dim=-1)

        return largest_eigenvalue(
            hvp,
            nb * h * n_items,
            dim,
            steps,
            seed=seed,
            dtype=weights[0].dtype,
            device=weights[0].device,
        )


def model_overshoot(
    model: SequenceModel,
    inputs: Tensor,
    iterations: int = 64,
    batch_chunk: int = 4,
    token_chunk: int = 16,
    seed: int = 0,
) -> dict[int, OvershootResult]:
    """mixer_overshoot for every TTT layer of model on token inputs [B, T], keyed by layer
    index. Each mixer is measured on the input it receives in the model's forward pass."""
    captured: dict[int, Tensor] = {}
    handles = []
    for layer, mixer in enumerate(model.mixers()):
        if isinstance(mixer, TTTMixer):

            def hook(_module: torch.nn.Module, args: tuple[Tensor, ...], layer: int = layer) -> None:
                captured[layer] = args[0].detach()

            handles.append(mixer.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    results: dict[int, OvershootResult] = {}
    for layer, mixer in enumerate(model.mixers()):
        if isinstance(mixer, TTTMixer):
            results[layer] = mixer_overshoot(
                mixer, captured[layer], iterations, batch_chunk, token_chunk, seed
            )
    return results
