"""Review guard for statespacexp-96a027: an explicit TTTOptions.eta_max overrides the
per-inner-model default (1.0 linear, 0.1 mlp). Without this test, a resolved_eta_max that
ignored the explicit value would pass the whole suite."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch import Tensor

from sst.models import ModelSpec, TTTOptions, build_mixer
from sst.models.mixers import TTTMixer


def _mixer(options: TTTOptions) -> TTTMixer:
    torch.manual_seed(0)
    mixer = build_mixer(ModelSpec("ttt", d_model=32, n_heads=2, options=options))
    assert isinstance(mixer, TTTMixer)
    return mixer


@pytest.mark.parametrize(
    ("inner_model", "default", "explicit"),
    [("mlp", 0.1, 0.37), ("mlp", 0.1, 1.0), ("linear", 1.0, 0.25)],
)
def test_explicit_eta_max_overrides_the_default(
    inner_model: Literal["linear", "mlp"], default: float, explicit: float
) -> None:
    implicit = _mixer(TTTOptions(inner_model=inner_model, minibatch=4))
    chosen = _mixer(TTTOptions(inner_model=inner_model, minibatch=4, eta_max=explicit))
    assert implicit.eta_max == default
    assert chosen.eta_max == explicit
    x = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out_implicit: Tensor = implicit(x)
        out_chosen: Tensor = chosen(x)
    # Same weights (same seed), different step size: the outputs must differ.
    assert (out_implicit - out_chosen).abs().max() > 1e-4


def test_explicit_eta_max_equal_to_the_default_changes_nothing() -> None:
    x = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(1))
    implicit = _mixer(TTTOptions(inner_model="mlp", minibatch=4))
    explicit = _mixer(TTTOptions(inner_model="mlp", minibatch=4, eta_max=0.1))
    with torch.no_grad():
        torch.testing.assert_close(explicit(x), implicit(x), atol=0.0, rtol=0.0)
