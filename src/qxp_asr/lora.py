"""Minimal LoRA (Hu et al., 2021, arXiv:2106.09685) for fairseq2 ``Linear`` projections, plus
activation checkpointing for fairseq2 transformer layers.

fairseq2 modules are not ``torch.nn.Linear`` subclasses, so off-the-shelf PEFT does not apply;
the wrapper below adds a trainable low-rank update ``B·A`` (B zero-initialised) on top of any
module exposing a 2-D ``weight``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LoRA(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        out_dim, in_dim = base.weight.shape
        self.base = base
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, in_dim, device=device))
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.drop(x).to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + (self.scale * update).to(x.dtype)


def add_lora(
    model: nn.Module, pattern: str, rank: int, alpha: float, dropout: float = 0.05
) -> list[str]:
    """Wrap every sub-module whose qualified name fully matches ``pattern``; return their names."""
    regex = re.compile(pattern)
    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if regex.fullmatch(name)
        and getattr(mod, "weight", None) is not None
        and mod.weight.dim() == 2
    ]
    for name, mod in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRA(mod, rank, alpha, dropout))
    return [n for n, _ in targets]


def checkpoint_layers(layers: Iterable[nn.Module]) -> None:
    """Recompute each layer's activations in backward (trades ~30 % compute for memory)."""
    for layer in layers:
        original = layer.forward

        def wrapped(*args, _f=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_f, *args, use_reentrant=False, **kwargs)
            return _f(*args, **kwargs)

        layer.forward = wrapped
