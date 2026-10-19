"""Geometry of the encoder LoRA updates: size, effective rank, and reproducibility across runs.

For every adapted projection of the encoder, the update is dW = (alpha / r) B A (alpha = 2r). Per
layer and module type we report

* relative size ||dW||_F / ||W||_F (W: the pretrained weight);
* effective rank exp(H(p)), p_i = s_i^2 / sum s^2 over dW's singular values (Roy & Vetterli, 2007);
* subspace overlap between two runs: ||U1^T U2||_F^2 / k for the top-k left singular vectors (the
  output directions a layer writes into), as in the LoRA paper's cross-seed analysis (Hu et al.,
  2022); a random k-dimensional subspace of R^d gives about k / d.

Writes ``reports/lora_geometry.csv`` (one row per layer and module) and prints per-quarter means.

    uv run python scripts/lora_geometry.py --run llm7b_lora_encdec --run llm7b_lora_encdec_s1 \
        --run llm7b_lora_enc
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import torch
import typer
from fairseq2.models.hub import load_model

from qxp_asr.paths import ARTIFACTS

KEY = re.compile(r"encoder\.layers\.(\d+)\.(.+)\.lora_a$")


def deltas(run: str) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """Thin SVD of every encoder update dW = 2 B A (alpha / r = 2) without forming dW:
    B = Qb Rb, A^T = Qa Ra, so dW = Qb (2 Rb Ra^T) Qa^T and the r x r core gives U, S."""
    st = torch.load(ARTIFACTS / "runs" / run / "best_trainable.pt", map_location="cpu",
                    weights_only=True)  # fmt: skip
    out = {}
    for k, a in st.items():
        m = KEY.fullmatch(k)
        if not m:
            continue
        b = st[k[: -len("lora_a")] + "lora_b"].float()
        qb, rb = torch.linalg.qr(b)
        _, ra = torch.linalg.qr(a.float().T)  # right factor only: U and S need just R_a
        u, s, _ = torch.linalg.svd(2.0 * rb @ ra.T)
        out[(int(m.group(1)), m.group(2))] = (qb @ u, s)  # left singular vectors, values
    return out


def eff_rank(s: torch.Tensor) -> float:
    p = s**2 / (s**2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def overlap(d1, d2, k: int) -> float:
    u1, u2 = d1[0][:, :k], d2[0][:, :k]
    return float((u1.T @ u2).pow(2).sum() / k)


def main(
    run: list[str] = typer.Option(...),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    k: int = typer.Option(4, help="subspace dimension for the overlap"),
    out: Path = typer.Option(Path("reports/lora_geometry.csv")),
) -> None:
    ds = {r: deltas(r) for r in run}
    model = load_model(card, device=torch.device("cpu"), dtype=torch.bfloat16)
    base = dict(model.named_parameters())
    rows = []
    for (layer, mod), d in sorted(ds[run[0]].items()):
        w_norm = float(base[f"encoder.layers.{layer}.{mod}.weight"].float().norm())
        u, s = d
        row = {
            "layer": layer,
            "module": mod,
            "rel_size": float(s.norm()) / w_norm,
            "eff_rank": eff_rank(s),
            "random_overlap": k / u.shape[0],
        }
        for other in run[1:]:
            if (layer, mod) in ds[other]:
                row[f"overlap_{other}"] = overlap(d, ds[other][(layer, mod)], k)
        rows.append(row)
    df = pl.DataFrame(rows)
    df.write_csv(out)
    n_layers = df["layer"].max() + 1
    q = df.with_columns(quarter=(pl.col("layer") * 4 // n_layers))
    cols = [c for c in df.columns if c not in ("layer", "module")]
    with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(f"reference run {run[0]}; per encoder quarter (0 = bottom):")
        print(q.group_by("quarter").agg(pl.col(cols).mean()).sort("quarter"))
        print("per module type:")
        print(df.group_by("module").agg(pl.col(cols).mean()).sort("module"))
    print(f"-> {out}; {n_layers} encoder layers")


if __name__ == "__main__":
    typer.run(main)
