"""The shared backbone. Only the sequence mixer differs between model families:

    embedding -> n_layers x [x = x + Mixer(RMSNorm(x)); x = x + MLP(RMSNorm(x))]
              -> RMSNorm -> linear readout
"""

from __future__ import annotations

from torch import Tensor, nn

from sst.models.layers import MLP, RMSNorm
from sst.models.mixers import Mixer, build_mixer
from sst.models.spec import ModelSpec


class Block(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.mixer_norm = RMSNorm(spec.d_model)
        self.mixer: Mixer = build_mixer(spec)
        self.mlp_norm = RMSNorm(spec.d_model)
        self.mlp = MLP(spec.d_model)

    def forward(self, x: Tensor) -> Tensor:
        mixed: Tensor = self.mixer(self.mixer_norm(x))
        x = x + mixed
        fed: Tensor = self.mlp(self.mlp_norm(x))
        return x + fed


class SequenceModel(nn.Module):
    """forward(inputs LongTensor [B, T]) -> logits FloatTensor [B, T, num_classes], causal."""

    def __init__(self, spec: ModelSpec, vocab_size: int, num_classes: int) -> None:
        super().__init__()
        self.spec = spec
        self.embedding = nn.Embedding(vocab_size, spec.d_model)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(spec.n_layers))
        self.final_norm = RMSNorm(spec.d_model)
        self.readout = nn.Linear(spec.d_model, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        x: Tensor = self.embedding(inputs)
        for block in self.blocks:
            x = block(x)
        logits: Tensor = self.readout(self.final_norm(x))
        return logits

    def mixers(self) -> list[Mixer]:
        return [block.mixer for block in self.blocks if isinstance(block, Block)]


def build_model(spec: ModelSpec, vocab_size: int, num_classes: int) -> SequenceModel:
    return SequenceModel(spec, vocab_size, num_classes)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
