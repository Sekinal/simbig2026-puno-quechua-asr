"""Paired bootstrap over clips: is system A's WER really different from system B's?

Acceptance protocol of reports/rl_experiment_plan.md: a change counts only if the 95 % interval
of the WER difference excludes 0. Clips are resampled with replacement (``--resamples`` times);
WER is micro-averaged (word edits / reference words) as in the official scorer.

    uv run python scripts/paired_bootstrap.py --a hybrid_dev_aug --b hybrid_dev_in
"""

from __future__ import annotations

import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from qxp_asr.metrics import official_normalize
from qxp_asr.paths import ARTIFACTS, MANIFESTS

SETS = {
    "dev_unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
    "itest_unseen": (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"),
    "dev_spont": pl.col("role") == "dev_spont",
    "itest_spont": pl.col("role") == "itest_spont",
    "ood": pl.col("role") == "ood",
}


def edits(refs: list[str], hyps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    r = [official_normalize(t).split() for t in refs]
    e = [
        Levenshtein.distance(x, official_normalize(h or "").split())
        for x, h in zip(r, hyps, strict=True)
    ]
    return np.array(e), np.array([len(x) for x in r])


def main(
    a: str = typer.Option(..., help="preds parquet of the new system"),
    b: str = typer.Option(..., help="preds parquet of the baseline"),
    sets: str = typer.Option(",".join(SETS)),
    resamples: int = typer.Option(2000),
    seed: int = typer.Option(0),
    cluster: bool = typer.Option(
        False,
        help="resample speakers (spontaneous) and sentences (read) instead of clips; OOD has no "
        "speaker labels and stays clip-level",
    ),
) -> None:
    pa = pl.read_parquet(ARTIFACTS / "preds" / f"{a}.parquet")
    pb = pl.read_parquet(ARTIFACTS / "preds" / f"{b}.parquet").select("clip_id", hyp_b="hyp")
    both = pa.join(pb, on="clip_id").filter(pl.col("text").is_not_null())
    spk = pl.read_parquet(MANIFESTS / "all.parquet").select("clip_id", "speaker")
    both = both.join(spk, on="clip_id", how="left")
    rng = np.random.default_rng(seed)
    rows = []
    for name in sets.split(","):
        d = both.filter(SETS[name])
        if d.is_empty():
            continue
        ea, n = edits(d["text"].to_list(), d["hyp"].to_list())
        eb, _ = edits(d["text"].to_list(), d["hyp_b"].to_list())
        if cluster and name != "ood":  # sum edits per cluster, then resample clusters
            key = d["text"] if "unseen" in name else d["speaker"]
            codes = key.rank("dense").cast(pl.Int64).to_numpy() - 1
            ea, eb, n = (np.bincount(codes, weights=x) for x in (ea, eb, n))
        idx = rng.integers(0, len(n), (resamples, len(n)))
        diff = 100 * (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(
            {
                "set": name,
                "units": len(n),
                "changed": int((d["hyp"] != d["hyp_b"]).sum()),
                "baseline": round(100 * eb.sum() / n.sum(), 2),
                "new": round(100 * ea.sum() / n.sum(), 2),
                "diff": round(100 * (ea.sum() - eb.sum()) / n.sum(), 2),
                "ci_lo": round(float(lo), 2),
                "ci_hi": round(float(hi), 2),
                "p_worse": round(float(np.mean(diff > 0)), 3),
            }
        )
    with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=200):
        print(f"{a} vs {b}")
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    typer.run(main)
