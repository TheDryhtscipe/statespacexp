"""Building blocks shared by every model: RMSNorm, the MLP, the short convolution and
the gated output path of the recurrent mixers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NORM_EPS = 1e-6


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight over the last dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + NORM_EPS) * self.weight


class MLP(nn.Module):
    """GELU MLP with 4x expansion: Linear(d, 4d) -> GELU -> Linear(4d, d)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, 4 * d_model)
        self.down = nn.Linear(4 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.up(x)
        out: Tensor = self.down(F.gelu(h))
        return out


class ShortConv(nn.Module):
    """Causal depthwise 1D convolution (kernel 4) over time, as in Mamba and DeltaNet.

    Input and output [B, T, C]. Left padding of kernel_size - 1 zeros keeps it causal.
    """

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        padded = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        out: Tensor = self.conv(padded)
        return out.transpose(1, 2)


class GatedOutput(nn.Module):
    """Output path shared by all recurrent mixers (GLA / Mamba-2 style):

        out = W_o( RMSNorm_per_head(o) * SiLU(W_g x) )

    o is the mixer's [B, T, d_model] output (heads concatenated) and x the mixer input.
    The per-head RMSNorm can be switched off with use_norm=False.
    """

    def __init__(self, d_model: int, n_heads: int, use_norm: bool) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.norm_weight = nn.Parameter(torch.ones(d_model)) if use_norm else None
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, o: Tensor, x: Tensor) -> Tensor:
        if self.norm_weight is not None:
            b, t, d = o.shape
            heads = o.view(b, t, self.n_heads, d // self.n_heads)
            heads = heads * torch.rsqrt(heads.pow(2).mean(dim=-1, keepdim=True) + NORM_EPS)
            o = heads.reshape(b, t, d) * self.norm_weight
        gate: Tensor = self.gate_proj(x)
        out: Tensor = self.out_proj(o * F.silu(gate))
        return out
