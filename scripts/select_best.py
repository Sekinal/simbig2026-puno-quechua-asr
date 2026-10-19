"""Print the prediction set with the best dev score (mean of unseen-scripted and spont WER).

Spontaneous dev uses clips <= 39 s (test clips are <= 25.8 s). Used by the final run script to pick
recipes automatically, so the choice is made on dev only and is reproducible.
"""

from __future__ import annotations

import polars as pl
import typer

from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS


def dev_score(name: str, man: pl.DataFrame) -> float:
    df = pl.read_parquet(ARTIFACTS / "preds" / f"{name}.parquet").drop(
        "duration_s", "unseen_sentence", "text", "role", strict=False
    )
    df = df.join(man, on="clip_id")
    unseen = df.filter((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"))
    spont = df.filter((pl.col("role") == "dev_spont") & (pl.col("duration_s") <= 39))
    a = wer_cer(unseen["text"].to_list(), unseen["hyp"].fill_null("").to_list())["wer"]
    b = wer_cer(spont["text"].to_list(), spont["hyp"].fill_null("").to_list())["wer"]
    return (a + b) / 2


def main(candidates: list[str]) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").select(
        "clip_id", "role", "unseen_sentence", "text", "duration_s"
    )
    scores = {c: dev_score(c, man) for c in candidates}
    for c, s in scores.items():
        print(f"# {c}: {s:.3f}", flush=True)
    print(min(scores, key=scores.get))


if __name__ == "__main__":
    typer.run(main)
