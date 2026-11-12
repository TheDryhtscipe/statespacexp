"""Overshoot ratio: per token, eta_t times the largest (algebraic) eigenvalue of the Hessian
of l_t(W) = 1/2 ||f_W(k_t) - v_t||^2 at the weights its gradient is taken at, and per
mini-batch the largest eigenvalue of sum_s eta_s Hess l_s(W_block_start).

References here are independent of the code under test: block-start weights from a
token-by-token loop with autograd gradients, and dense Hessians from
torch.autograd.functional.hessian with torch.linalg.eigvalsh.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sst.overshoot import (
    OvershootResult,
    largest_eigenvalue,
    mixer_overshoot,
    model_overshoot,
)
from sst.models import ModelSpec, TTTOptions, build_mixer, build_model
from sst.models.mixers import TTTMixer
from sst.models.model import Block
from sst.models.ttt import InnerModel, TTTScanInputs, inner_forward


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --- the eigenvalue solver -------------------------------------------------------------


def _matrix_hvp(mats: Tensor) -> Callable[[Tensor], Tensor]:
    return lambda vec: (mats @ vec.unsqueeze(-1)).squeeze(-1)


def _symmetric(n_items: int, n: int, seed: int) -> Tensor:
    a = torch.randn(n_items, n, n, generator=_gen(seed), dtype=torch.float64)
    return 0.5 * (a + a.transpose(-1, -2))


def test_largest_eigenvalue_is_algebraic_for_indefinite_matrices() -> None:
    """Dominant negative eigenvalue: power iteration on H would return it (in magnitude);
    the largest algebraic eigenvalue is the most positive one."""
    n = 30
    g = _gen(1)
    basis = torch.linalg.qr(torch.randn(4, n, n, generator=g, dtype=torch.float64)).Q
    spectrum = torch.linspace(-0.2, 0.2, n, dtype=torch.float64).repeat(4, 1)
    spectrum[:, 0] = -10.0
    spectrum[:, -1] = torch.tensor([1.0, 0.5, -0.1, 0.0], dtype=torch.float64)  # top
    spectrum[2, 1:-1] = -0.5 - torch.rand(n - 2, generator=g, dtype=torch.float64)  # all < 0
    spectrum[3, 1:-1] = -torch.rand(n - 2, generator=g, dtype=torch.float64)
    mats = basis @ torch.diag_embed(spectrum) @ basis.transpose(-1, -2)
    exact = torch.linalg.eigvalsh(mats)[:, -1]
    assert (exact.abs() < torch.linalg.eigvalsh(mats)[:, 0].abs()).all()
    got = largest_eigenvalue(_matrix_hvp(mats), 4, n, iterations=n, seed=0)
    torch.testing.assert_close(got.value, exact, atol=1e-9, rtol=1e-9)


def test_largest_eigenvalue_converges_with_fewer_iterations_than_dimension() -> None:
    mats = _symmetric(8, 200, seed=2)
    exact = torch.linalg.eigvalsh(mats)[:, -1]
    got = largest_eigenvalue(_matrix_hvp(mats), 8, 200, iterations=120, seed=0)
    torch.testing.assert_close(got.value, exact, atol=1e-6, rtol=1e-6)
    assert (got.residual < 1e-4).all()


def test_largest_eigenvalue_handles_low_rank_breakdown() -> None:
    """Rank one (Krylov space invariant after one step) and zero matrices."""
    n = 12
    u = torch.randn(3, n, generator=_gen(3), dtype=torch.float64)
    mats = -2.0 * u.unsqueeze(-1) * u.unsqueeze(-2)  # negative semidefinite: top is 0
    mats[1] = 3.0 * torch.outer(u[1], u[1])  # positive: top is 3 ||u||^2
    mats[2] = 0.0
    got = largest_eigenvalue(_matrix_hvp(mats), 3, n, iterations=8, seed=0)
    expected = torch.tensor([0.0, 3.0 * float(u[1].pow(2).sum()), 0.0], dtype=torch.float64)
    torch.testing.assert_close(got.value, expected, atol=1e-9, rtol=1e-9)


# --- the overshoot ratio on TTT mixers --------------------------------------------------


def _mixer(
    inner_model: InnerModel,
    inner_norm: bool,
    minibatch: int,
    token_eta: bool = False,
    normalise_keys: bool = True,
) -> TTTMixer:
    """A tiny mixer (head dim 4) in float64, with non-trivial W_0 and large values so the
    inner residuals, and with them the indefinite Hessian terms, are not small."""
    torch.manual_seed(0)
    options = TTTOptions(
        inner_model=inner_model,
        inner_norm=inner_norm,
        minibatch=minibatch,
        token_eta=token_eta,
        normalise_keys=normalise_keys,
        eta_max=0.5,
    )
    mixer = build_mixer(ModelSpec("ttt", d_model=8, n_heads=2, options=options))
    assert isinstance(mixer, TTTMixer)
    mixer.double()
    with torch.no_grad():
        mixer.v_proj.weight.mul_(5.0)
        for p in mixer.inner_init:
            p.add_(0.5 * torch.randn_like(p))
        if mixer.learnable_token_idx is not None:
            mixer.learnable_token_idx.copy_(torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64))
    return mixer


def _x(b: int = 2, t: int = 7) -> Tensor:
    return torch.randn(b, t, 8, generator=_gen(4), dtype=torch.float64)


def _flat_shapes(init: tuple[Tensor, ...]) -> tuple[list[torch.Size], list[int]]:
    return [p.shape[1:] for p in init], [p[0].numel() for p in init]


def _reference_starts(i: TTTScanInputs, mixer: TTTMixer) -> tuple[list[list[Tensor]], Tensor]:
    """Per (batch, head): the flattened block-start weights for every token, and the
    effective rate eta_t * s_c (the block-end scale with which the gradient is carried)."""
    b, h, t, _ = i.q.shape
    mb = mixer.cfg.minibatch
    shapes, sizes = _flat_shapes(i.init)
    starts: list[list[Tensor]] = []
    eff = torch.zeros(b, h, t, dtype=i.q.dtype)
    for bi in range(b):
        row: list[Tensor] = []
        for hi in range(h):
            w = torch.cat([p[hi].reshape(-1) for p in i.init]).detach()
            per_token: list[Tensor] = []
            for first in range(0, t, mb):
                last = min(first + mb, t)
                scale = 1.0 if i.token_scale is None else float(i.token_scale[last - first - 1])
                acc = torch.zeros_like(w)
                for s in range(first, last):
                    per_token.append(w)
                    leaf = w.clone().requires_grad_(True)
                    ws = tuple(
                        p.view(1, 1, 1, *sh) for p, sh in zip(torch.split(leaf, sizes), shapes, strict=True)
                    )
                    nrm = None if i.norm is None else (i.norm[0][hi : hi + 1], i.norm[1][hi : hi + 1])
                    k = i.k[bi, hi, s].view(1, 1, 1, -1)
                    v = i.v[bi, hi, s].view(1, 1, 1, -1)
                    loss = 0.5 * (inner_forward(ws, k, mixer.cfg, nrm) - v).pow(2).sum()
                    acc = acc + i.eta[bi, hi, s] * torch.autograd.grad(loss, leaf)[0]
                    eff[bi, hi, s] = i.eta[bi, hi, s] * scale
                w = (w - scale * acc).detach()
            row.append(torch.stack(per_token))
        starts.append(row)
    return starts, eff


def _token_hessian(i: TTTScanInputs, mixer: TTTMixer, w: Tensor, bi: int, hi: int, s: int) -> Tensor:
    shapes, sizes = _flat_shapes(i.init)
    nrm = None if i.norm is None else (i.norm[0][hi : hi + 1], i.norm[1][hi : hi + 1])
    k = i.k[bi, hi, s].view(1, 1, 1, -1).detach()
    v = i.v[bi, hi, s].view(1, 1, 1, -1).detach()

    def loss(flat: Tensor) -> Tensor:
        ws = tuple(p.view(1, 1, 1, *sh) for p, sh in zip(torch.split(flat, sizes), shapes, strict=True))
        return 0.5 * (inner_forward(ws, k, mixer.cfg, nrm) - v).pow(2).sum()

    hess = torch.autograd.functional.hessian(loss, w)  # type: ignore[no-untyped-call]
    assert isinstance(hess, Tensor)
    return hess


def _reference(mixer: TTTMixer, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(eta_eff, curvature, ratio, block_ratio, most negative eigenvalue) by dense Hessians."""
    with torch.no_grad():
        i = mixer.scan_inputs(x)
    b, h, t, _ = i.q.shape
    mb = mixer.cfg.minibatch
    n_blocks = -(-t // mb)
    starts, eff = _reference_starts(i, mixer)
    curvature = torch.zeros(b, h, t, dtype=x.dtype)
    lowest = torch.zeros(b, h, t, dtype=x.dtype)
    block = torch.zeros(b, h, n_blocks, dtype=x.dtype)
    for bi in range(b):
        for hi in range(h):
            for j in range(n_blocks):
                total = torch.zeros(())
                for s in range(j * mb, min((j + 1) * mb, t)):
                    hess = _token_hessian(i, mixer, starts[bi][hi][s], bi, hi, s)
                    eig = torch.linalg.eigvalsh(hess)
                    curvature[bi, hi, s], lowest[bi, hi, s] = eig[-1], eig[0]
                    total = total + eff[bi, hi, s] * hess
                block[bi, hi, j] = torch.linalg.eigvalsh(total)[-1]
    return eff, curvature, eff * curvature, block, lowest


def _check(got: OvershootResult, mixer: TTTMixer, x: Tensor) -> Tensor:
    eff, curvature, ratio, block, lowest = _reference(mixer, x)
    torch.testing.assert_close(got.eta, eff, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(got.curvature, curvature, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got.ratio, ratio, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got.block_ratio, block, atol=1e-6, rtol=1e-6)
    return lowest


@pytest.mark.parametrize("minibatch", [1, 3])
def test_linear_without_norm_curvature_is_squared_key_norm(minibatch: int) -> None:
    """H = (k k^T) kron I, so lambda_max = ||k||^2 exactly, and the block Hessian is
    (sum_s eta_s k_s k_s^T) kron I."""
    mixer = _mixer("linear", inner_norm=False, minibatch=minibatch, normalise_keys=False)
    x = _x()
    got = mixer_overshoot(mixer, x)
    with torch.no_grad():
        i = mixer.scan_inputs(x)
    k_sq = i.k.pow(2).sum(-1)
    assert (k_sq - 1.0).abs().max() > 0.1  # unnormalised keys, so this is not trivially 1
    torch.testing.assert_close(got.curvature, k_sq, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(got.ratio, i.eta * k_sq, atol=1e-8, rtol=1e-8)
    b, h, t, _ = i.q.shape
    expected_block = torch.zeros(b, h, -(-t // minibatch), dtype=x.dtype)
    for j in range(expected_block.shape[-1]):
        ks = i.k[:, :, j * minibatch : (j + 1) * minibatch]
        es = i.eta[:, :, j * minibatch : (j + 1) * minibatch]
        gram = torch.einsum("bhs,bhsi,bhsj->bhij", es, ks, ks)
        expected_block[:, :, j] = torch.linalg.eigvalsh(gram)[..., -1]
    torch.testing.assert_close(got.block_ratio, expected_block, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    ("inner_model", "inner_norm", "minibatch", "token_eta"),
    [
        ("mlp", False, 1, False),
        ("mlp", False, 3, False),
        ("linear", True, 1, False),
        ("linear", True, 3, True),
        ("mlp", True, 3, True),
    ],
)
def test_ratios_match_dense_hessian_eigenvalues(
    inner_model: InnerModel, inner_norm: bool, minibatch: int, token_eta: bool
) -> None:
    mixer = _mixer(inner_model, inner_norm, minibatch, token_eta)
    x = _x()  # T = 7: the last mini-batch of 3 is a single token
    got = mixer_overshoot(mixer, x)
    lowest = _check(got, mixer, x)
    if inner_norm:
        # The case a naive power iteration gets wrong: the negative eigenvalue dominates.
        assert ((-lowest > got.curvature) & (got.curvature > 0)).any()


def test_chunking_does_not_change_the_result() -> None:
    mixer = _mixer("mlp", inner_norm=True, minibatch=3, token_eta=True)
    x = _x(b=3)
    whole = mixer_overshoot(mixer, x, batch_chunk=8, token_chunk=64)
    pieces = mixer_overshoot(mixer, x, batch_chunk=1, token_chunk=2)
    for a, e in zip(pieces, whole, strict=True):
        torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-6)


def test_model_overshoot_measures_each_ttt_layer_on_its_own_input() -> None:
    torch.manual_seed(0)
    spec = ModelSpec(
        "ttt", d_model=8, n_heads=2, n_layers=2, options=TTTOptions(inner_model="mlp", minibatch=2)
    )
    model = build_model(spec, vocab_size=5, num_classes=3).double()
    inputs = torch.randint(0, 5, (2, 6), generator=_gen(5))
    got = model_overshoot(model, inputs)
    assert sorted(got) == [0, 1]
    with torch.no_grad():
        x = model.embedding(inputs)
        for layer, block in enumerate(model.blocks):
            assert isinstance(block, Block)
            mixer = block.mixer
            assert isinstance(mixer, TTTMixer)
            expected = mixer_overshoot(mixer, block.mixer_norm(x))
            for a, e in zip(got[layer], expected, strict=True):
                torch.testing.assert_close(a, e)
            x = block(x)
    assert not any(p.grad is not None for p in model.parameters())
