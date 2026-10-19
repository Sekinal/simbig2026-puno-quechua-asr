import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lora_surgery import SCALE, pad, svd_factors, truncate


def _lora(out_dim=40, in_dim=30, r=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out_dim, r, generator=g), torch.randn(r, in_dim, generator=g)


def test_svd_factors_reconstruct_update():
    b, a = _lora()
    u, s, v = svd_factors(b, a)
    assert torch.allclose((u * s) @ v.T, SCALE * b.double() @ a.double(), atol=1e-8)


def test_full_rank_truncation_is_exact():
    b, a = _lora()
    b2, a2 = truncate(b, a, 6)
    assert torch.allclose(b2 @ a2, b @ a, atol=1e-4)


def test_truncation_is_best_low_rank_approximation():
    b, a = _lora()
    b2, a2 = truncate(b, a, 2)
    dw = SCALE * b.double() @ a.double()
    s = torch.linalg.svdvals(dw)
    err = torch.linalg.norm(dw - SCALE * (b2 @ a2).double())
    assert torch.isclose(err, s[2:].norm(), atol=1e-3)


def test_concatenated_average_equals_mean_update():
    (b1, a1), (b2, a2) = _lora(seed=1), _lora(seed=2)
    b, a = torch.cat([b1, b2], 1) / 2, torch.cat([a1, a2], 0)
    assert torch.allclose(b @ a, (b1 @ a1 + b2 @ a2) / 2, atol=1e-5)


def test_pad_keeps_update():
    b, a = _lora(r=4)
    b2, a2 = pad(b, a, 9)
    assert a2.shape[0] == 9
    assert torch.allclose(b2 @ a2, b @ a)
