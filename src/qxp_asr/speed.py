"""Numerically safe throughput settings shared by all trainers.

* TF32 for any remaining fp32 matmuls/convolutions (bf16 autocast already covers the hot path);
* cuDNN autotuning (input lengths vary, but the CNN front-end sees a bounded set of shapes);
* fused AdamW (single CUDA kernel per step; identical update rule).
None of these changes the optimisation problem.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def enable_fast_math() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def adamw(params: Iterable, **kwargs) -> torch.optim.AdamW:
    """AdamW with the fused CUDA implementation when every parameter lives on the GPU."""
    groups = list(params)
    on_cuda = all(p.is_cuda for g in groups for p in (g["params"] if isinstance(g, dict) else [g]))
    return torch.optim.AdamW(groups, fused=on_cuda, **kwargs)
