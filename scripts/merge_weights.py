"""Weight-space merging of qxp CTC checkpoints: WiSE-FT interpolation and checkpoint soups.

* WiSE-FT (Wortsman et al., 2022, CVPR, arXiv:2109.01903): theta = (1-a)*theta_zs + a*theta_ft
  keeps most in-distribution gains while improving robustness under distribution shift — our
  test has new speakers, new sentences, elderly voices and uncontrolled devices.
* Model soups / checkpoint averaging (Wortsman et al., 2022, ICML, arXiv:2203.05482; Izmailov et
  al., 2018, SWA, arXiv:1803.05407): uniform averaging of fine-tuned checkpoints along a run (or
  across seeds) flattens the solution and typically lowers error at no inference cost.

``--zeroshot-weight`` adds the pruned-head zero-shot omniASR weights as one more soup ingredient.

Run inside ``envs/omni`` (CPU is fine):
    uv run python /workspace/simbig/repo/scripts/merge_weights.py --out wise_a0.8.pt \
        --ckpt /workspace/simbig/artifacts/runs/ctc1b_v2/best.pt:0.8 --zeroshot-weight 0.2
"""

from __future__ import annotations

import torch
import typer

from qxp_asr.omni import load_qxp_ctc
from qxp_asr.paths import ARTIFACTS


def main(
    out: str = typer.Option(..., help="output file name under artifacts/merged"),
    ckpt: list[str] = typer.Option(..., help="path[:weight] (repeatable)"),
    zeroshot_weight: float = typer.Option(0.0),
    card: str = typer.Option("omniASR_CTC_1B_v2"),
) -> None:
    parts: list[tuple[dict[str, torch.Tensor], float]] = []
    for spec in ckpt:
        path, _, w = spec.rpartition(":") if ":" in spec else (spec, "", "1")
        parts.append((torch.load(path, map_location="cpu", weights_only=True), float(w)))
    if zeroshot_weight > 0:
        zs = load_qxp_ctc(card, None, device="cpu", dtype=torch.float32).state_dict()
        parts.append((zs, zeroshot_weight))
    total = sum(w for _, w in parts)
    merged = {
        k: sum(sd[k].float() * (w / total) for sd, w in parts).to(parts[0][0][k].dtype)
        for k in parts[0][0]
    }
    dst = ARTIFACTS / "merged" / out
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, dst)
    print(f"merged {len(parts)} state dicts (weights {[round(w / total, 3) for _, w in parts]})")
    print(dst)


if __name__ == "__main__":
    typer.run(main)
