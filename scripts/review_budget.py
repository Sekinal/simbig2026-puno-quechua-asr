"""Human-in-the-loop budget: how much does a speaker checking only flagged words reduce WER?

For speech communities without transcribers, a native speaker can often *check* words faster than
transcribe. Simulation: the word-level verifier (``word_verifier.py --dump``; cross-fitted
P(correct) of each word slot of the hybrid output) ranks the output's word slots from least to most
likely correct; a reviewer checks the top ``f`` share of slots and fixes every wrong one (each wrong
slot is one word error: substitution, insertion or deletion). Reported per set: WER after review as
a function of ``f`` for verifier-ranked, random and oracle (perfect error detection) review order,
and the number of words reviewed per hour of audio.

    uv run python scripts/review_budget.py --preds wordver_dev_in --feats wordfeat_dev_in
"""

from __future__ import annotations

import numpy as np
import polars as pl
import typer

from qxp_asr.paths import ARTIFACTS, MANIFESTS

SETS = {"dev_spont": "dev spontaneous", "itest_spont": "itest spontaneous", "ood": "OOD radio"}
FRACS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def main(
    preds: str = typer.Option(...),
    feats: str = typer.Option(..., help="word_features output (for the reference word counts)"),
    out: str = typer.Option("review_budget"),
    seed: int = typer.Option(0),
) -> None:
    p = pl.read_parquet(ARTIFACTS / "cands" / f"{preds}.parquet").filter(pl.col("in_pivot") == 1)
    clips = pl.read_parquet(ARTIFACTS / "cands" / f"{feats}_clips.parquet")
    dur = pl.read_parquet(MANIFESTS / "all.parquet").select("clip_id", "duration_s")
    rng = np.random.default_rng(seed)
    rows = []
    for role, name in SETS.items():
        g = p.filter(pl.col("role") == role)
        c = clips.filter(pl.col("role") == role).join(dur, on="clip_id")
        ref_words = sum(len(r.split()) for r in c["ref"])
        hours = c["duration_s"].sum() / 3600
        wrong = 1 - g["label"].to_numpy()
        n = len(wrong)
        orders = {
            "verifier": np.argsort(g["p"].to_numpy()),
            "random": rng.permutation(n),
            "oracle": np.argsort(-wrong, kind="stable"),
        }
        for f in FRACS:
            k = round(f * n)
            row = {"set": name, "reviewed_share": f, "words_per_audio_hour": round(k / hours)}
            for how, order in orders.items():
                fixed = wrong[order[:k]].sum()
                row[how] = round(100 * (wrong.sum() - fixed) / ref_words, 2)
            rows.append(row)
    res = pl.DataFrame(rows)
    res.write_csv(f"reports/{out}.csv")
    with pl.Config(tbl_rows=40, tbl_cols=-1):
        print(res)


if __name__ == "__main__":
    typer.run(main)
