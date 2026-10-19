"""Print WER/CER rows (LaTeX-ready) for prediction tables on the paper's evaluation sets.

uv run python scripts/ladder.py zs_llm7b ctc1b_v3_soup_wise0.85_greedy hybrid_dev_aug
"""

from __future__ import annotations

import polars as pl
import typer

from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS

SETS = {
    "dev unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
    "itest unseen": (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"),
    "dev spont": pl.col("role") == "dev_spont",
    "itest spont": pl.col("role") == "itest_spont",
    "OOD": pl.col("role") == "ood",
}


def main(preds: list[str]) -> None:
    print("system & " + " & ".join(SETS) + r" \\")
    for name in preds:
        d = pl.read_parquet(ARTIFACTS / "preds" / f"{name}.parquet").filter(
            pl.col("text").is_not_null()
        )
        cells = []
        for cond in SETS.values():
            g = d.filter(cond)
            if g.is_empty():
                cells.append("--")
                continue
            s = wer_cer(g["text"].to_list(), g["hyp"].to_list())
            cells.append(f"{s['wer']:.1f}/{s['cer']:.1f}")
        print(f"{name} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    typer.run(main)
