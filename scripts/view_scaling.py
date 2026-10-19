"""Test-time scaling curve: WER vs the number of perturbed views added to the candidate pool.

Uses a pool built with ``llm_candidates --views K`` (per-view source tags ``<adapter>@aug<j>``).
For k = 0..K, keep the candidates proposed by beam search / CTC or by views j < k, recompute the
view-agreement feature on those views only, select with fixed tuned hybrid weights, and report per
set the hybrid WER and the pool oracle WER. A curve that keeps falling is a test-time-compute
scaling result; the gap between the two curves is the verification headroom.

    uv run python scripts/view_scaling.py --cands pool_dev_aug --weights hybrid_dev_aug.weights.json
"""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from joint_rescore2 import build_features
from qxp_asr.metrics import official_normalize, wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS

SETS = {
    "dev_unseen": ((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"), "scripted"),
    "itest_unseen": ((pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"), "scripted"),
    "dev_spont": (pl.col("role") == "dev_spont", "spont"),
    "itest_spont": (pl.col("role") == "itest_spont", "spont"),
    "ood": (pl.col("role") == "ood", "spont"),
}
AUG = re.compile(r"@aug(\d+)$")


def main(
    cands: str = typer.Option(...),
    weights: str = typer.Option(...),
    post: str = typer.Option("ctc1b_v3_soup_wise0.85"),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("indomain"),
    workers: int = typer.Option(8),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    saved = json.loads((ARTIFACTS / "cands" / weights).read_text())
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("text").is_not_null() & (pl.col("duration_s") <= 39)
    )
    srcs = {(c, t): s for c, t, s in table.select("clip_id", "text", "sources").iter_rows()}
    n_views = 1 + max(
        (int(m.group(1)) for s in table["sources"].to_list() for x in s if (m := AUG.search(x))),
        default=-1,
    )
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    rows = []
    for name, (cond, domain) in SETS.items():
        sub = man.filter(cond)
        feats = build_features(table, sub["clip_id"].to_list(), post, char_lm, workers, False, True)
        names = ["chr", "len", *lp_cols, "naug"]
        w = np.array([saved[domain].get(k, 0.0) for k in names])
        for k in range(n_views + 1):
            refs, hyb, orc = [], [], []
            for cid, ref in sub.select("clip_id", "text").iter_rows():
                if cid not in feats:
                    continue
                texts, f, _ = feats[cid]
                keep, naug = [], []
                for i, t in enumerate(texts):
                    tags = srcs[(cid, t)]
                    views = [int(m.group(1)) for x in tags if (m := AUG.search(x))]
                    base = any(AUG.search(x) is None for x in tags)
                    if base or any(v < k for v in views):
                        keep.append(i)
                        naug.append(sum(v < k for v in views))
                g = f[keep].copy()
                g[:, -1] = naug  # agreement over the views in use only
                score = g[:, 0] + g[:, 1:] @ w
                r = official_normalize(ref)
                kept = [texts[i] for i in keep]
                errs = [
                    Levenshtein.distance(r.split(), official_normalize(t).split()) for t in kept
                ]
                refs.append(ref)
                hyb.append(kept[int(np.argmax(score))])
                orc.append(kept[int(np.argmin(errs))])
            rows.append(
                {
                    "set": name,
                    "views": k,
                    "hybrid": round(wer_cer(refs, hyb)["wer"], 2),
                    "oracle": round(wer_cer(refs, orc)["wer"], 2),
                }
            )
        print(pl.DataFrame(rows).filter(pl.col("set") == name), flush=True)
    res = pl.DataFrame(rows)
    res.write_csv(f"reports/view_scaling_{cands}.csv")
    with pl.Config(tbl_rows=60, tbl_cols=-1):
        print(res.pivot(on="set", index="views", values="hybrid"))
        print(res.pivot(on="set", index="views", values="oracle"))


if __name__ == "__main__":
    typer.run(main)
