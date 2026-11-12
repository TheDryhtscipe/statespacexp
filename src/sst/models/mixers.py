"""Sequence mixers. Each maps x [B, T, d_model] to [B, T, d_model] causally.

The recurrent matrix-state mixers share one front end and one output path, so that
only the transition A_t differs between them:

  front end:  q, k, v = SiLU(ShortConv(W_{q,k,v} x))   (conv optional, kernel 4)
              per head dk = dv = d_model / n_heads
              delta family and TTT: q, k L2-normalised; others: q scaled by dk^-1/2
  core:       S_t = S_{t-1} A_t + (write),  o_t = S_t q_t      (see cores.py)
  output:     W_o(RMSNorm_per_head(o) * SiLU(W_g x))           (see layers.GatedOutput)

Decay parametrisation shared by diag_pos, diag_neg and the radius of diag_complex:
alpha_t = exp(-softplus(W x_t + b)) in (0, 1), with b set so that alpha at zero input is
uniform in [ALPHA_INIT_MIN, ALPHA_INIT_MAX]. diag_neg maps alpha' = 2 alpha - 1 and
initialises alpha in [(1 + ALPHA_INIT_MIN) / 2, (1 + ALPHA_INIT_MAX) / 2], so that its
alpha' at zero input has the same distribution as diag_pos's alpha. Likewise delta_neg
(and deltaproduct with allow_negative) initialise the beta bias so that beta at zero
input equals the (0, 1) variant's. Each pair then differs in reachable range only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sst.models.cores import (
    complex_diag_scan,
    delta_scan,
    diag_scan,
    householder_matrices,
    householder_scan,
    linear_attention,
)
from sst.models.layers import GatedOutput, ShortConv
from sst.models.s4d import S4DLayer
from sst.models.spec import (
    DeltaProductOptions,
    ModelSpec,
    S4DOptions,
    TransformerOptions,
    TTTOptions,
)
from sst.models.ttt import TTTInnerConfig, TTTScanInputs, ttt_scan

ALPHA_INIT_MIN = 0.5
ALPHA_INIT_MAX = 0.99
L2_EPS = 1e-6
INITIAL_STATE_STD = 0.02


class Mixer(nn.Module):
    """Base class: forward(x [B, T, D]) -> [B, T, D]."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def realised_transitions(self, x: Tensor) -> Tensor:
        """The transition matrices A_t [B, H, T, dk, dk] this mixer applies for input x."""
        raise NotImplementedError(f"{type(self).__name__} has no matrix transition")


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    """[B, T, H * d] -> [B, H, T, d]."""
    b, t, hd = x.shape
    return x.view(b, t, n_heads, hd // n_heads).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    """[B, H, T, d] -> [B, T, H * d]."""
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


def _decay_bias(size: int, negative: bool = False) -> Tensor:
    """Bias b with exp(-softplus(b)) uniform in [ALPHA_INIT_MIN, ALPHA_INIT_MAX], or, for
    negative=True, with 2 exp(-softplus(b)) - 1 uniform in that range."""
    alpha0 = torch.empty(size).uniform_(ALPHA_INIT_MIN, ALPHA_INIT_MAX)
    if negative:
        alpha0 = (alpha0 + 1.0) / 2.0
    rate = -torch.log(alpha0)  # softplus(b) = rate
    return torch.log(torch.expm1(rate))


def _decay_projection(d_model: int, out: int, negative: bool = False) -> nn.Linear:
    proj = nn.Linear(d_model, out)
    with torch.no_grad():
        proj.bias.copy_(_decay_bias(out, negative))
    return proj


class RecurrentMixer(Mixer):
    """Shared front end and output path for the matrix-state linear RNNs.

    n_steps > 1 gives n_steps separate (k, v) pairs per token (DeltaProduct).
    scale_q multiplies unnormalised queries by dk^-1/2 (dot-product readouts).
    matrix_state: the core keeps S in R^{dv x dk}, so learn_initial_state adds a learned
    per-head S_0 (initial_state, [H, dv, dk]).
    """

    def __init__(
        self,
        spec: ModelSpec,
        normalise_qk: bool,
        n_steps: int = 1,
        scale_q: bool = True,
        matrix_state: bool = True,
    ) -> None:
        super().__init__()
        d = spec.d_model
        self.n_heads = spec.n_heads
        self.head_dim = spec.head_dim
        self.n_steps = n_steps
        self.normalise_qk = normalise_qk
        self.scale_q = scale_q
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, n_steps * d, bias=False)
        self.v_proj = nn.Linear(d, n_steps * d, bias=False)
        self.q_conv = ShortConv(d) if spec.use_short_conv else None
        self.k_conv = ShortConv(n_steps * d) if spec.use_short_conv else None
        self.v_conv = ShortConv(n_steps * d) if spec.use_short_conv else None
        self.output = GatedOutput(d, spec.n_heads, spec.output_norm)
        self.initial_state = (
            nn.Parameter(torch.randn(spec.n_heads, spec.head_dim, spec.head_dim) * INITIAL_STATE_STD)
            if spec.learn_initial_state and matrix_state
            else None
        )

    def s0(self) -> Tensor | None:
        """Learned initial state [1, H, dv, dk], or None (S_0 = 0)."""
        return None if self.initial_state is None else self.initial_state.unsqueeze(0)

    @staticmethod
    def _feature(proj: nn.Linear, conv: ShortConv | None, x: Tensor) -> Tensor:
        h: Tensor = proj(x)
        if conv is not None:
            h = conv(h)
        return F.silu(h)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """q [B, H, T, dk], k [B, H, T, n_steps, dk], v [B, H, T, n_steps, dv]."""
        b, t, _ = x.shape
        h, d, n = self.n_heads, self.head_dim, self.n_steps
        q = _split_heads(self._feature(self.q_proj, self.q_conv, x), h)
        k = self._feature(self.k_proj, self.k_conv, x).view(b, t, n, h, d).permute(0, 3, 1, 2, 4)
        v = self._feature(self.v_proj, self.v_conv, x).view(b, t, n, h, d).permute(0, 3, 1, 2, 4)
        if self.normalise_qk:
            q = F.normalize(q, dim=-1, eps=L2_EPS)
            k = F.normalize(k, dim=-1, eps=L2_EPS)
        elif self.scale_q:
            q = q * self.head_dim**-0.5
        return q, k, v

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Per-head outputs [B, H, T, dv]."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.project(x)
        o = self.core(x, q, k, v)
        out: Tensor = self.output(_merge_heads(o), x)
        return out

    def _per_head(self, proj: nn.Linear, x: Tensor, n_steps: int = 1) -> Tensor:
        """proj(x) [B, T, n_steps * H * m] -> [B, H, T, n_steps, m] (squeezed if n_steps == 1)."""
        b, t, _ = x.shape
        y: Tensor = proj(x)
        y = y.view(b, t, n_steps, self.n_heads, -1).permute(0, 3, 1, 2, 4)
        return y.squeeze(3) if n_steps == 1 else y

    def _batch_eye(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        eye = torch.eye(self.head_dim, dtype=x.dtype, device=x.device)
        return eye.expand(b, self.n_heads, t, self.head_dim, self.head_dim)


class LinAttnMixer(RecurrentMixer):
    """A_t = I, write v k^T (computed in the parallel causal form)."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, normalise_qk=False)

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        out = linear_attention(q, k.squeeze(3), v.squeeze(3))
        if self.initial_state is not None:
            # A_t = I, so S_t = S_0 + sum_{s <= t} v_s k_s^T and S_0 adds S_0 q_t.
            out = out + torch.einsum("hvk,bhtk->bhtv", self.initial_state, q)
        return out

    def realised_transitions(self, x: Tensor) -> Tensor:
        return self._batch_eye(x)


class DiagFixedMixer(RecurrentMixer):
    """A = diag(tanh(a)), a learned per head and key dim, independent of the input."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, normalise_qk=False)
        alpha0 = torch.empty(spec.n_heads, spec.head_dim).uniform_(ALPHA_INIT_MIN, ALPHA_INIT_MAX)
        self.alpha_raw = nn.Parameter(torch.atanh(alpha0))

    def alpha(self) -> Tensor:
        """[1, H, 1, dk]."""
        return torch.tanh(self.alpha_raw)[None, :, None, :]

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return diag_scan(q, k.squeeze(3), v.squeeze(3), self.alpha(), self.s0()).outputs

    def realised_transitions(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        alpha = self.alpha().expand(b, self.n_heads, t, self.head_dim)
        return torch.diag_embed(alpha)


class DiagGatedMixer(RecurrentMixer):
    """A_t = diag(alpha_t), alpha_t = exp(-softplus(W x_t + b)) per head and key dim.

    negative=False: alpha in (0, 1) (diag_pos, GLA / Mamba-2 style).
    negative=True: alpha' = 2 alpha - 1 in (-1, 1) (diag_neg, Grazzi et al. 2025).
    Write term v k^T, not scaled by the step size.
    """

    def __init__(self, spec: ModelSpec, negative: bool) -> None:
        super().__init__(spec, normalise_qk=False)
        self.negative = negative
        self.alpha_proj = _decay_projection(spec.d_model, spec.d_model, negative)

    def alpha(self, x: Tensor) -> Tensor:
        """[B, H, T, dk]."""
        alpha = torch.exp(-F.softplus(self._per_head(self.alpha_proj, x)))
        return 2.0 * alpha - 1.0 if self.negative else alpha

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return diag_scan(q, k.squeeze(3), v.squeeze(3), self.alpha(x), self.s0()).outputs

    def realised_transitions(self, x: Tensor) -> Tensor:
        return torch.diag_embed(self.alpha(x))


class DiagComplexMixer(RecurrentMixer):
    """Key dims paired into complex coordinates; A_t = blockdiag(r_t R(theta_t)).

    r_t = exp(-softplus(W_r x_t + b_r)) in (0, 1) and theta_t = W_th x_t + b_th (radians,
    unbounded; b_th initialised to 0), per head and key-dim pair.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, normalise_qk=False)
        self.radius_proj = _decay_projection(spec.d_model, spec.d_model // 2)
        self.angle_proj = nn.Linear(spec.d_model, spec.d_model // 2)
        nn.init.zeros_(self.angle_proj.bias)

    def radius_angle(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Each [B, H, T, dk / 2]."""
        radius = torch.exp(-F.softplus(self._per_head(self.radius_proj, x)))
        return radius, self._per_head(self.angle_proj, x)

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        radius, angle = self.radius_angle(x)
        return complex_diag_scan(
            q, k.squeeze(3), v.squeeze(3), radius, angle, self.s0()
        ).outputs

    def realised_transitions(self, x: Tensor) -> Tensor:
        radius, angle = self.radius_angle(x)
        c, s = radius * torch.cos(angle), radius * torch.sin(angle)
        blocks = torch.stack((torch.stack((c, s), -1), torch.stack((-s, c), -1)), -2)  # [..., P, 2, 2]
        b, h, t, p = radius.shape
        a = x.new_zeros(b, h, t, 2 * p, 2 * p)
        for i in range(p):
            a[..., 2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = blocks[..., i, :, :]
        return a


class HouseholderMixer(RecurrentMixer):
    """Delta rule and DeltaProduct.

    Per token, n_steps generalised Householder updates
    S <- S (I - beta_j k_j k_j^T) + beta_j v_j k_j^T with k_j L2-normalised and
    beta_j = beta_max * sigmoid(w_j . x), beta_max = 2 if allow_negative else 1.
    use_delta_core=True runs delta_scan (n_steps must be 1); otherwise householder_scan.
    """

    def __init__(
        self, spec: ModelSpec, n_steps: int, allow_negative: bool, use_delta_core: bool
    ) -> None:
        super().__init__(spec, normalise_qk=True, n_steps=n_steps)
        if use_delta_core and n_steps != 1:
            raise ValueError("the delta core takes one step per token")
        self.beta_max = 2.0 if allow_negative else 1.0
        self.use_delta_core = use_delta_core
        self.beta_proj = nn.Linear(spec.d_model, n_steps * spec.n_heads)
        if allow_negative:
            # Same beta at zero input as the (0, 1) variant: 2 sigmoid(b') = sigmoid(b).
            with torch.no_grad():
                self.beta_proj.bias.copy_(torch.logit(torch.sigmoid(self.beta_proj.bias) / 2.0))

    def beta(self, x: Tensor) -> Tensor:
        """[B, H, T, n_steps]."""
        b, t, _ = x.shape
        logits: Tensor = self.beta_proj(x)
        beta = self.beta_max * torch.sigmoid(logits)
        return beta.view(b, t, self.n_steps, self.n_heads).permute(0, 3, 1, 2)

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        beta = self.beta(x)
        if self.use_delta_core:
            return delta_scan(q, k.squeeze(3), v.squeeze(3), beta.squeeze(3), self.s0()).outputs
        return householder_scan(q, k, v, beta, self.s0()).outputs

    def realised_transitions(self, x: Tensor) -> Tensor:
        _, k, _ = self.project(x)
        house = householder_matrices(k, self.beta(x))  # [B, H, T, n, dk, dk]
        a = house[:, :, :, 0]
        for j in range(1, self.n_steps):
            a = a @ house[:, :, :, j]
        return a


class TTTMixer(RecurrentMixer):
    """Test-time training layer (see ttt.py). eta_t = eta_max * sigmoid(w . x_t) per head.
    With token_eta, the readout at mini-batch position i uses the weights
    W_start - s_i * sum_{r <= i} eta_r grad_r, where the accumulated update (not the
    readout) is scaled by s_i = max(1/i + learnable_token_idx[i-1], 0) (see ttt_scan).

    The initial inner weights W_0 (and the inner LN) are learned, shared across
    sequences. normalise_keys L2-normalises both keys and queries.
    """

    token_idx: Tensor

    def __init__(self, spec: ModelSpec, options: TTTOptions) -> None:
        super().__init__(
            spec, normalise_qk=options.normalise_keys, scale_q=False, matrix_state=False
        )
        self.cfg = TTTInnerConfig(options.inner_model, options.inner_norm, options.minibatch)
        self.eta_max = options.resolved_eta_max()
        self.eta_proj = nn.Linear(spec.d_model, spec.n_heads)
        h, d = spec.n_heads, spec.head_dim
        self.inner_init = nn.ParameterList()
        if options.inner_model == "linear":
            self.inner_init.append(nn.Parameter(torch.randn(h, d, d) * 0.02))
        else:
            self.inner_init.append(nn.Parameter(torch.randn(h, 4 * d, d) * 0.02))
            self.inner_init.append(nn.Parameter(torch.zeros(h, 4 * d)))
            self.inner_init.append(nn.Parameter(torch.randn(h, d, 4 * d) * 0.02))
            self.inner_init.append(nn.Parameter(torch.zeros(h, d)))
        self.ln_weight = nn.Parameter(torch.ones(h, d)) if options.inner_norm else None
        self.ln_bias = nn.Parameter(torch.zeros(h, d)) if options.inner_norm else None
        # Published token_eta: 1/i for i = 1..minibatch plus a learned offset (init 0).
        self.register_buffer(
            "token_idx", 1.0 / torch.arange(1, options.minibatch + 1), persistent=False
        )
        self.learnable_token_idx = (
            nn.Parameter(torch.zeros(options.minibatch)) if options.token_eta else None
        )

    def inner_rates(self, x: Tensor) -> Tensor:
        """Per-token inner learning rate eta [B, H, T] (before any token_eta scaling)."""
        return self.eta_max * torch.sigmoid(self._per_head(self.eta_proj, x).squeeze(-1))

    def token_scale(self) -> Tensor | None:
        """token_eta: s_i = max(1/i + offset_i, 0) for i = 1..minibatch; else None."""
        if self.learnable_token_idx is None:
            return None
        return torch.clamp_min(self.token_idx + self.learnable_token_idx, 0.0)

    def inner_norm(self) -> tuple[Tensor, Tensor] | None:
        if self.ln_weight is None or self.ln_bias is None:
            return None
        return self.ln_weight, self.ln_bias

    def scan_inputs(self, x: Tensor) -> TTTScanInputs:
        """Everything ttt_scan receives for mixer input x [B, T, d_model]."""
        q, k, v = self.project(x)
        return self._scan_inputs(x, q, k, v)

    def _scan_inputs(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> TTTScanInputs:
        return TTTScanInputs(
            q=q,
            k=k.squeeze(3),
            v=v.squeeze(3),
            eta=self.inner_rates(x),
            init=tuple(p for p in self.inner_init),
            norm=self.inner_norm(),
            token_scale=self.token_scale(),
        )

    def core(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        i = self._scan_inputs(x, q, k, v)
        scan = ttt_scan(i.q, i.k, i.v, i.eta, i.init, self.cfg, i.norm, token_scale=i.token_scale)
        return scan.outputs


class S4DMixer(Mixer):
    """u = SiLU(ShortConv(W_in x)); y = GELU(S4D(u)); then the shared gated output path."""

    def __init__(self, spec: ModelSpec, options: S4DOptions) -> None:
        super().__init__()
        d = spec.d_model
        self.in_proj = nn.Linear(d, d, bias=False)
        self.conv = ShortConv(d) if spec.use_short_conv else None
        self.ssm = S4DLayer(
            d, options.state_size, options.dt_min, options.dt_max, spec.learn_initial_state
        )
        self.output = GatedOutput(d, spec.n_heads, spec.output_norm)

    def forward(self, x: Tensor) -> Tensor:
        u: Tensor = self.in_proj(x)
        if self.conv is not None:
            u = self.conv(u)
        y: Tensor = self.ssm(F.silu(u))
        out: Tensor = self.output(F.gelu(y), x)
        return out


def _rope(x: Tensor, base: float = 10000.0) -> Tensor:
    """Rotary position embedding on [B, H, T, d] (rotate-half convention)."""
    t, d = x.shape[2], x.shape[3]
    half = d // 2
    freqs = base ** (-torch.arange(half, dtype=x.dtype, device=x.device) / half)
    angles = torch.arange(t, dtype=x.dtype, device=x.device)[:, None] * freqs[None, :]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class AttentionMixer(Mixer):
    """Causal softmax multi-head attention with RoPE or no positional encoding."""

    def __init__(self, spec: ModelSpec, options: TransformerOptions) -> None:
        super().__init__()
        self.n_heads = spec.n_heads
        self.use_rope = options.pos == "rope"
        self.qkv_proj = nn.Linear(spec.d_model, 3 * spec.d_model, bias=False)
        self.out_proj = nn.Linear(spec.d_model, spec.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        qkv: Tensor = self.qkv_proj(x)
        q, k, v = (_split_heads(part, self.n_heads) for part in qkv.chunk(3, dim=-1))
        if self.use_rope:
            q, k = _rope(q), _rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out: Tensor = self.out_proj(_merge_heads(o))
        return out


class LSTMMixer(Mixer):
    """torch nn.LSTM (hidden = d_model) followed by a linear output projection.

    learn_initial_state adds learned (h_0, c_0), each [1, 1, d_model].
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        d = spec.d_model
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.out_proj = nn.Linear(d, d, bias=False)
        learn = spec.learn_initial_state
        self.initial_h = nn.Parameter(torch.randn(1, 1, d) * INITIAL_STATE_STD) if learn else None
        self.initial_c = nn.Parameter(torch.randn(1, 1, d) * INITIAL_STATE_STD) if learn else None

    def forward(self, x: Tensor) -> Tensor:
        if self.initial_h is not None and self.initial_c is not None:
            b = x.shape[0]
            hc = (
                self.initial_h.expand(1, b, -1).contiguous(),
                self.initial_c.expand(1, b, -1).contiguous(),
            )
            h: Tensor = self.lstm(x, hc)[0]
        else:
            h = self.lstm(x)[0]
        out: Tensor = self.out_proj(h)
        return out


def build_mixer(spec: ModelSpec) -> Mixer:
    options = spec.resolved_options()
    kind = spec.mixer
    if kind == "lin_attn":
        return LinAttnMixer(spec)
    if kind == "diag_fixed":
        return DiagFixedMixer(spec)
    if kind == "diag_pos":
        return DiagGatedMixer(spec, negative=False)
    if kind == "diag_neg":
        return DiagGatedMixer(spec, negative=True)
    if kind == "diag_complex":
        return DiagComplexMixer(spec)
    if kind == "delta":
        return HouseholderMixer(spec, 1, allow_negative=False, use_delta_core=True)
    if kind == "delta_neg":
        return HouseholderMixer(spec, 1, allow_negative=True, use_delta_core=True)
    if kind == "deltaproduct" and isinstance(options, DeltaProductOptions):
        return HouseholderMixer(
            spec, options.n_householder, options.allow_negative, use_delta_core=False
        )
    if kind == "ttt" and isinstance(options, TTTOptions):
        return TTTMixer(spec, options)
    if kind == "s4d" and isinstance(options, S4DOptions):
        return S4DMixer(spec, options)
    if kind == "transformer" and isinstance(options, TransformerOptions):
        return AttentionMixer(spec, options)
    if kind == "lstm":
        return LSTMMixer(spec)
    raise ValueError(f"cannot build mixer {kind!r} with options {options!r}")

