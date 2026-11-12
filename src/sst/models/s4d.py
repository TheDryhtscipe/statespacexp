"""S4D: a diagonal, linear time-invariant SSM (Gu et al. 2022, "On the Parameterization
and Initialization of Diagonal State Space Models").

Per channel d (one SSM per model channel) with N/2 complex modes n:

    x_t = lambda x_{t-1} + b_bar u_t,    y_t = 2 Re(sum_n c_n x_{t,n}) + D u_t,

lambda = exp(dt A) and b_bar = (lambda - 1) / A * B (zero-order hold). The modes come
in conjugate pairs, which the factor 2 Re(.) accounts for, so state_size N means N/2
stored complex modes (a real state of dimension N). Nothing depends on the input
except u itself, so the map u -> y is a causal convolution with kernel
K[l] = 2 Re(sum_n c_n b_bar_n lambda_n^l), computed here with an FFT.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def s4d_kernel(lam: Tensor, b_bar: Tensor, c: Tensor, length: int) -> Tensor:
    """Vandermonde kernel K[d, l] = 2 Re(sum_n c[d, n] b_bar[d, n] lam[d, n]^l), real [D, L]."""
    steps = torch.arange(length, device=lam.device, dtype=lam.real.dtype)
    powers = torch.exp(torch.log(lam).unsqueeze(-1) * steps)  # [D, N2, L]
    return 2.0 * torch.einsum("dn,dnl->dl", c * b_bar, powers).real


def fft_causal_conv(u: Tensor, kernel: Tensor) -> Tensor:
    """y[b, t, d] = sum_{s <= t} kernel[d, t - s] u[b, s, d] via FFT.

    Both signals are zero-padded to length 2T so the circular convolution of the FFT
    equals the linear (causal) one on the first T outputs.
    """
    t = u.shape[1]
    n = 2 * t
    u_f = torch.fft.rfft(u.transpose(1, 2), n=n)  # [B, D, n/2+1]
    k_f = torch.fft.rfft(kernel[:, :t], n=n)  # [D, n/2+1]
    y: Tensor = torch.fft.irfft(u_f * k_f, n=n)[..., :t]
    return y.transpose(1, 2)


def s4d_initial_response(lam: Tensor, c: Tensor, x0: Tensor, length: int) -> Tensor:
    """Output contribution of an initial state x0 (the state before the first token).

    x_t = lam^{t+1} x0 + (input terms), so the contribution is
    2 Re(sum_n c_n lam_n^{t+1} x0_n) for t = 0..length-1. Returns real [length, D].
    """
    steps = torch.arange(1, length + 1, device=lam.device, dtype=lam.real.dtype)
    powers = torch.exp(torch.log(lam).unsqueeze(-1) * steps)  # [D, N2, L]
    return 2.0 * torch.einsum("dn,dnl->ld", c * x0, powers).real


def s4d_recurrence(
    u: Tensor, lam: Tensor, b_bar: Tensor, c: Tensor, initial_state: Tensor | None = None
) -> Tensor:
    """The same map as fft_causal_conv(u, s4d_kernel(...)) (plus s4d_initial_response
    when initial_state [D, N/2] is given), as an explicit loop over time."""
    b, t_len, d = u.shape
    x = torch.zeros(b, d, lam.shape[1], dtype=lam.dtype, device=u.device)
    if initial_state is not None:
        x = x + initial_state
    ys: list[Tensor] = []
    for t in range(t_len):
        x = lam * x + b_bar * u[:, t].unsqueeze(-1)
        ys.append(2.0 * (c * x).sum(dim=-1).real)
    return torch.stack(ys, dim=1)


class S4DLayer(nn.Module):
    """Learned S4D parameters with the S4D-Lin initialisation.

    A_n = -exp(a_re_log) + i a_im, initialised to -1/2 + i pi n. dt per channel is
    log-uniform in [dt_min, dt_max]. B is initialised to 1 and C to a standard complex
    normal; both are trained and neither depends on the input. D is a per-channel skip.
    learn_initial_state adds a learned initial state x_0 (N(0, 0.02) per part).
    Complex parameters are stored as pairs of real tensors.
    """

    def __init__(
        self,
        channels: int,
        state_size: int,
        dt_min: float,
        dt_max: float,
        learn_initial_state: bool = False,
    ) -> None:
        super().__init__()
        if state_size % 2 != 0:
            raise ValueError("state_size must be even (conjugate pairs)")
        modes = state_size // 2
        self.log_dt = nn.Parameter(
            torch.rand(channels) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.a_re_log = nn.Parameter(torch.full((channels, modes), math.log(0.5)))
        modes_n = torch.arange(modes, dtype=self.log_dt.dtype)
        self.a_im = nn.Parameter(math.pi * modes_n.expand(channels, modes).clone())
        self.b_re = nn.Parameter(torch.ones(channels, modes))
        self.b_im = nn.Parameter(torch.zeros(channels, modes))
        self.c_re = nn.Parameter(torch.randn(channels, modes) * math.sqrt(0.5))
        self.c_im = nn.Parameter(torch.randn(channels, modes) * math.sqrt(0.5))
        self.skip = nn.Parameter(torch.ones(channels))
        self.initial_state_re = (
            nn.Parameter(torch.randn(channels, modes) * 0.02) if learn_initial_state else None
        )
        self.initial_state_im = (
            nn.Parameter(torch.randn(channels, modes) * 0.02) if learn_initial_state else None
        )

    def discretise(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (lambda, b_bar, c), each complex [D, N/2]."""
        a = torch.complex(-torch.exp(self.a_re_log), self.a_im)
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        lam = torch.exp(dt * a)
        b = torch.complex(self.b_re, self.b_im)
        b_bar = (lam - 1.0) / a * b
        return lam, b_bar, torch.complex(self.c_re, self.c_im)

    def forward(self, u: Tensor) -> Tensor:
        """u: [B, T, D] -> y: [B, T, D]."""
        lam, b_bar, c = self.discretise()
        kernel = s4d_kernel(lam, b_bar, c, u.shape[1])
        y = fft_causal_conv(u, kernel) + self.skip * u
        if self.initial_state_re is not None and self.initial_state_im is not None:
            x0 = torch.complex(self.initial_state_re, self.initial_state_im)
            y = y + s4d_initial_response(lam, c, x0, u.shape[1])
        return y
