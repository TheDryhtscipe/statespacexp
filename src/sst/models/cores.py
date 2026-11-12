"""Reference recurrences for the matrix-state linear RNN family.

Every scan here is a plain loop over time, written to be obviously correct rather than
fast. Per head the state is S_t in R^{dv x dk} and

    S_t = S_{t-1} A_t + (write term),    o_t = S_t q_t,

so the output at step t reads the state *after* the token-t update. Tensor layout:
q, k: [B, H, T, dk]; v: [B, H, T, dv]; states: [B, H, dv, dk].
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class ScanResult(NamedTuple):
    """Outputs [B, H, T, dv], final state [B, H, dv, dk], optional states [B, H, T, dv, dk]."""

    outputs: Tensor
    final_state: Tensor
    states: Tensor | None


def _initial_state(q: Tensor, v: Tensor, initial_state: Tensor | None) -> Tensor:
    b, h, _, dk = q.shape
    dv = v.shape[-1]
    if initial_state is None:
        return q.new_zeros(b, h, dv, dk)
    return initial_state.expand(b, h, dv, dk)


def _read(state: Tensor, q_t: Tensor) -> Tensor:
    """o = S q for S [B, H, dv, dk], q [B, H, dk]."""
    return torch.einsum("bhvk,bhk->bhv", state, q_t)


def _outer(v_t: Tensor, k_t: Tensor) -> Tensor:
    """v k^T for v [..., dv], k [..., dk]."""
    return v_t.unsqueeze(-1) * k_t.unsqueeze(-2)


def _steps(x: Tensor) -> tuple[Tensor, ...]:
    """Split [B, H, T, ...] into T tensors [B, H, ...].

    unbind backpropagates with one stack; indexing x[:, :, t] inside the loop would
    allocate a full-size zero gradient for every step.
    """
    return x.unbind(dim=2)


def _finish(outputs: list[Tensor], state: Tensor, states: list[Tensor] | None) -> ScanResult:
    return ScanResult(
        outputs=torch.stack(outputs, dim=2),
        final_state=state,
        states=None if states is None else torch.stack(states, dim=2),
    )


def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal linear attention in parallel form: o_t = sum_{s <= t} (q_t . k_s) v_s.

    Equal to the recurrence S_t = S_{t-1} + v_t k_t^T, o_t = S_t q_t (A_t = I), but
    computed without a loop, so it serves as an independent check on the scans.
    """
    t = q.shape[2]
    scores = q @ k.transpose(-1, -2)  # [B, H, T, T], scores[t, s] = q_t . k_s
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    return scores.masked_fill(~mask, 0.0) @ v


def diag_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    initial_state: Tensor | None = None,
    return_states: bool = False,
) -> ScanResult:
    """S_t = S_{t-1} diag(alpha_t) + v_t k_t^T.

    alpha broadcasts to [B, H, T, dk] (an input-independent transition passes
    [1, H, 1, dk]).
    """
    alpha = alpha.expand_as(k)
    state = _initial_state(q, v, initial_state)
    outputs: list[Tensor] = []
    states: list[Tensor] | None = [] if return_states else None
    for q_t, k_t, v_t, a_t in zip(_steps(q), _steps(k), _steps(v), _steps(alpha), strict=True):
        state = state * a_t.unsqueeze(-2) + _outer(v_t, k_t)
        outputs.append(_read(state, q_t))
        if states is not None:
            states.append(state)
    return _finish(outputs, state, states)


def complex_diag_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    radius: Tensor,
    angle: Tensor,
    initial_state: Tensor | None = None,
    return_states: bool = False,
) -> ScanResult:
    """Diagonal-over-C transition: key dims (2i, 2i+1) form one complex coordinate.

    A_t is block-diagonal with 2x2 blocks r [[cos th, sin th], [-sin th, cos th]],
    so each row (a, b) of a column pair of S maps to r (a cos - b sin, a sin + b cos):
    a rotation by +theta scaled by r. Block eigenvalues are r e^{+-i theta}.
    radius, angle: [B, H, T, dk / 2]. Write term v_t k_t^T.
    """
    b, h, t_len, dk = q.shape
    dv = v.shape[-1]
    radius = radius.expand(b, h, t_len, dk // 2)
    angle = angle.expand(b, h, t_len, dk // 2)
    cos = radius * torch.cos(angle)
    sin = radius * torch.sin(angle)
    state = _initial_state(q, v, initial_state)
    outputs: list[Tensor] = []
    states: list[Tensor] | None = [] if return_states else None
    steps = zip(_steps(q), _steps(k), _steps(v), _steps(cos), _steps(sin), strict=True)
    for q_t, k_t, v_t, cos_t, sin_t in steps:
        pairs = state.reshape(b, h, dv, dk // 2, 2)
        re, im = pairs[..., 0], pairs[..., 1]
        c_t = cos_t.unsqueeze(-2)
        s_t = sin_t.unsqueeze(-2)
        rotated = torch.stack((re * c_t - im * s_t, re * s_t + im * c_t), dim=-1)
        state = rotated.reshape(b, h, dv, dk) + _outer(v_t, k_t)
        outputs.append(_read(state, q_t))
        if states is not None:
            states.append(state)
    return _finish(outputs, state, states)


def delta_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    return_states: bool = False,
) -> ScanResult:
    """Delta rule in error-correction form: S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T.

    Equal to S_{t-1} (I - beta k k^T) + beta v k^T. beta: [B, H, T]. Keys are expected
    to be unit norm (the caller normalises).
    """
    state = _initial_state(q, v, initial_state)
    outputs: list[Tensor] = []
    states: list[Tensor] | None = [] if return_states else None
    for q_t, k_t, v_t, b_t in zip(_steps(q), _steps(k), _steps(v), _steps(beta), strict=True):
        error = _read(state, k_t) - v_t  # S k - v, [B, H, dv]
        state = state - b_t[..., None, None] * _outer(error, k_t)
        outputs.append(_read(state, q_t))
        if states is not None:
            states.append(state)
    return _finish(outputs, state, states)


def householder_matrices(k: Tensor, beta: Tensor) -> Tensor:
    """I - beta k k^T for k [..., dk], beta [...]: returns [..., dk, dk]."""
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    return eye - beta[..., None, None] * (k.unsqueeze(-1) * k.unsqueeze(-2))


def householder_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    return_states: bool = False,
) -> ScanResult:
    """DeltaProduct: n_h generalised Householder steps per token, applied in order.

    For j = 1..n_h: S <- S (I - beta_j k_j k_j^T) + beta_j v_j k_j^T, with the matrix
    materialised explicitly (a separate code path from delta_scan). Then o_t = S_t q_t.
    k: [B, H, T, n_h, dk]; v: [B, H, T, n_h, dv]; beta: [B, H, T, n_h].
    """
    state = _initial_state(q, v, initial_state)
    outputs: list[Tensor] = []
    states: list[Tensor] | None = [] if return_states else None
    house = householder_matrices(k, beta)  # [B, H, T, n_h, dk, dk]
    for q_t, k_t, v_t, b_t, house_t in zip(
        _steps(q), _steps(k), _steps(v), _steps(beta), _steps(house), strict=True
    ):
        steps = zip(k_t.unbind(2), v_t.unbind(2), b_t.unbind(2), house_t.unbind(2), strict=True)
        for k_j, v_j, b_j, house_j in steps:
            state = state @ house_j + b_j[..., None, None] * _outer(v_j, k_j)
        outputs.append(_read(state, q_t))
        if states is not None:
            states.append(state)
    return _finish(outputs, state, states)
