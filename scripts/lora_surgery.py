"""Rewrite trained LoRA adapters in weight space: spectral truncation and exact seed averaging.

Every adapted projection carries the update dW = s B A (s = alpha / r = 2 for all our adapters,
``qxp_asr.lora``). Two operations, applied per projection and written back as a LoRA state dict
that ``scripts/infer_llm.py`` loads with the new rank:

* several ``--run``: the exact mean of the runs' updates, (dW_1 + ... + dW_n) / n, is itself a LoRA
  of rank n r: B = [B_1, ..., B_n] / n, A = [A_1; ...; A_n].
* ``--top k``: the best rank-k approximation of the (averaged) update (Eckart-Young), from a thin
  SVD that never forms dW: B = Q_b R_b, A^T = Q_a R_a, so dW = Q_b (s R_b R_a^T) Q_a^T and the
  small core's SVD U S V^T gives dW = (Q_b U) S (Q_a V)^T; then B' = (Q_b U_k) S_k / s,
  A' = (Q_a V_k)^T.

Does the useful update live in a few directions (truncation), and is the part two seeds share
better than either seed (average, then truncate)? Truncation is applied to every adapted
projection (``--part``: encoder, decoder or both; the other part keeps its rank-r update, padded
with zeros to the output rank so one LoRA rank serves the whole model).

    uv run python scripts/lora_surgery.py --run llm7b_lora_enc --top 4 --out enc_top4
"""

from __future__ import annotations

import re

import torch
import typer

from qxp_asr.paths import ARTIFACTS

SCALE = 2.0  # alpha / rank, as in train_llm / infer_llm (alpha = 2 r)


def svd_factors(
    b: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left vectors, singular values, right vectors of SCALE * b @ a without forming it."""
    qb, rb = torch.linalg.qr(b.double())
    qa, ra = torch.linalg.qr(a.double().T)
    u, s, vh = torch.linalg.svd(SCALE * rb @ ra.T)
    return qb @ u, s, qa @ vh.T


def truncate(b: torch.Tensor, a: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, v = svd_factors(b, a)
    return (u[:, :k] * s[:k] / SCALE).float(), v[:, :k].T.float().contiguous()


def pad(b: torch.Tensor, a: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a rank-r0 LoRA to rank r (the update is unchanged)."""
    r0 = a.shape[0]
    if r0 == r:
        return b, a
    b2 = torch.zeros(b.shape[0], r, dtype=b.dtype)
    a2 = torch.zeros(r, a.shape[1], dtype=a.dtype)
    b2[:, :r0], a2[:r0] = b, a
    return b2, a2


def main(
    run: list[str] = typer.Option(..., help="run(s) under artifacts/runs (several: averaged)"),
    top: int = typer.Option(0, help="keep the top-k singular directions (0 = no truncation)"),
    part: str = typer.Option("encoder", help="where to truncate: encoder | decoder | both"),
    out: str = typer.Option(..., help="output run name (artifacts/runs/<out>/best_trainable.pt)"),
) -> None:
    states = [
        torch.load(
            ARTIFACTS / "runs" / r / "best_trainable.pt", map_location="cpu", weights_only=True
        )
        for r in run
    ]
    n = len(states)
    keys = [k for k in states[0] if k.endswith(".lora_a")]
    for st in states[1:]:
        if set(st) != set(states[0]):
            raise KeyError("runs adapt different parameters")
    pairs = {}
    for ka in keys:
        kb = ka[: -len("lora_a")] + "lora_b"
        a = torch.cat([st[ka].float() for st in states], 0)
        b = torch.cat([st[kb].float() for st in states], 1) / n
        is_enc = ka.startswith("encoder.")
        if top and (part == "both" or (part == "encoder") == is_enc):
            b, a = truncate(b, a, top)
        pairs[ka] = (b, a)
    rank = max(a.shape[0] for _, a in pairs.values())
    new = {}
    for ka, factors in pairs.items():
        b, a = pad(*factors, rank)
        new[ka], new[ka[: -len("lora_a")] + "lora_b"] = a, b
    for k in states[0]:  # non-LoRA trainables (none for our encoder / decoder runs): mean
        if not re.search(r"\.lora_[ab]$", k):
            new[k] = sum(st[k].float() for st in states) / n
    dest = ARTIFACTS / "runs" / out
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(new, dest / "best_trainable.pt")
    enc = [k for k in pairs if k.startswith("encoder.")]
    print(f"{out}: {len(pairs)} projections ({len(enc)} encoder), LoRA rank {rank}")
    print(f"load with: --encoder-lora {rank if enc else 0} "
          f"--decoder-lora {rank if len(enc) < len(pairs) else 0}")  # fmt: skip


if __name__ == "__main__":
    typer.run(main)
