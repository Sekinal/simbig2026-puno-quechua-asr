"""Reward benchmark: how well does each *reference-free* reward rank ASR candidates?

RL on unlabelled audio is only as good as its reward. Before spending GPU hours on GSPO/GRPO, we
score every candidate of a pooled N-best table (``scripts/llm_candidates.py``; ~12-16 candidates
per clip from CTC + several LLM-ASR adapters) with each candidate reward and compare the ranking
with the true error against the gold transcript (LITERATURE §g.11 and §g′, "reward bench").

Metrics per evaluation bucket (role; spontaneous also split by speaker age):

* ``pair_acc``  among candidate pairs of a clip whose true error differs (word edits, then char
  edits), the share the reward orders correctly (reward ties count 1/2). This is what a
  group-relative policy gradient (GRPO/GSPO) consumes: the within-group ordering.
* ``sel_wer``   corpus WER when the reward alone picks one candidate per clip.

Rewards: CTC log-likelihood (total and per character), char-10gram LM, each LLM adapter's
log-likelihood, the exp11 RL verifier reward (per-char CTC + 0.3 per-char char-LM), consensus
(negative mean distance to the other candidates), number of proposing systems; and, as references,
the supervised hybrid score (weights tuned on labelled dev) and the oracle.

    uv run python scripts/reward_bench.py --cands pool_dev_tc --post ctc1b_v3_soup_wise0.85 \
        --weights hybrid_dev_tc.weights.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from joint_rescore2 import build_features
from qxp_asr.metrics import official_normalize, wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS

BUCKETS = {"dev_unseen": "scripted", "dev_spont": "spont", "itest_spont": "spont", "ood": "spont"}
OLD = ["fifties", "sixties", "seventies", "eighties", "nineties"]


def rewards(
    texts: list[str],
    f: np.ndarray,
    risk: np.ndarray,
    lp_cols: list[str],
    w: dict,
    fst: dict[str, bool],
):
    """Candidate rewards (higher = better) from the joint_rescore2 feature matrix
    [ctc, chr, len, lp_*..., nsrc] and the pairwise distance matrix."""
    n = np.maximum(f[:, 2], 1)
    out = {
        "ctc": f[:, 0],
        "ctc/char": f[:, 0] / n,
        "chr_lm": f[:, 1],
        "chr_lm/char": f[:, 1] / n,
        "rl_verifier": np.maximum(f[:, 0] / n + 0.3 * f[:, 1] / n, -50.0),
        "consensus": -risk.mean(axis=1),
        "nsrc": f[:, -1],
    }
    if fst:  # share of words a Southern Quechua morphological FST accepts (scripts/fst_accept.py)
        out["fst_valid"] = np.array(
            [np.mean([fst.get(x, False) for x in t.split()]) if t else 0.0 for t in texts]
        )
        out["ctc+fst"] = f[:, 0] + 20 * out["fst_valid"]
    for j, c in enumerate(lp_cols):
        out[c] = f[:, 3 + j]
        out[f"{c}/char"] = f[:, 3 + j] / n
    if w:  # supervised reference: the tuned hybrid score
        lin = np.array([w["chr"], w["len"], *[w[c] for c in lp_cols]])
        out["hybrid(tuned)"] = f[:, 0] + f[:, 1 : 3 + len(lp_cols)] @ lin
    return out


def errors(ref: str, texts: list[str]) -> np.ndarray:
    """Word edits + 1e-3 char edits: orders by WER, ties broken by CER."""
    r = official_normalize(ref)
    return np.array(
        [
            Levenshtein.distance(r.split(), official_normalize(t).split())
            + 1e-3 * Levenshtein.distance(r, official_normalize(t))
            for t in texts
        ]
    )


def pair_stats(err: np.ndarray, rew: np.ndarray) -> tuple[float, int]:
    de = err[:, None] - err[None, :]
    dr = rew[:, None] - rew[None, :]
    mask = de < 0  # i strictly better than j
    if not mask.any():
        return 0.0, 0
    correct = (dr[mask] > 0).sum() + 0.5 * (dr[mask] == 0).sum()
    return float(correct), int(mask.sum())


def main(
    cands: str = typer.Option(...),
    post: str = typer.Option(...),
    weights: str = typer.Option("", help="tuned hybrid weights json (supervised reference)"),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("all", help="'indomain' = provided data only"),
    workers: int = typer.Option(8),
    out: str = typer.Option("", help="write the table to reports/<out>.csv"),
    fst: str = typer.Option("", help="word acceptance table from scripts/fst_accept.py"),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    man = pl.read_parquet(MANIFESTS / "all.parquet").with_columns(
        bucket=pl.when((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"))
        .then(pl.lit("dev_unseen"))
        .otherwise(pl.col("role"))
    )
    man = man.filter(pl.col("bucket").is_in(list(BUCKETS)) & pl.col("text").is_not_null())
    # only clips <= 39 s carry LLM candidates (the test set has no longer clips)
    man = man.filter(pl.col("duration_s") <= 39.0)
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    feats = build_features(table, man["clip_id"].to_list(), post, char_lm, workers, True)
    saved = json.loads((ARTIFACTS / "cands" / weights).read_text()) if weights else {}
    accept = pl.read_parquet(fst) if fst else None
    fst_ok = (
        dict(zip(accept["word"], accept["accepted"], strict=True)) if accept is not None else {}
    )
    acc: dict[tuple[str, str], list] = {}
    for cid, ref, bucket, age in man.select("clip_id", "text", "bucket", "age").iter_rows():
        if cid not in feats:
            continue
        texts, f, risk = feats[cid]
        err = errors(ref, texts)
        groups = [bucket] + (
            [f"{bucket}:{'60+' if age in OLD[1:] else '<60'}"]
            if bucket not in {"dev_unseen", "ood"}
            else []
        )
        rew = rewards(texts, f, risk, lp_cols, saved.get(BUCKETS[bucket], {}), fst_ok)
        rew["oracle"] = -err
        for name, r in rew.items():
            c, n = pair_stats(err, r)
            pick = texts[int(np.argmax(r))]
            for g in groups:
                a = acc.setdefault((g, name), [0.0, 0, [], []])
                a[0] += c
                a[1] += n
                a[2].append(ref)
                a[3].append(pick)
    rows = [
        {
            "bucket": g,
            "reward": name,
            "pair_acc": round(100 * c / max(1, n), 1),
            "sel_wer": round(wer_cer(refs, picks)["wer"], 2),
            "clips": len(refs),
            "pairs": n,
        }
        for (g, name), (c, n, refs, picks) in acc.items()
    ]
    res = pl.DataFrame(rows)
    wide = res.pivot(on="bucket", index="reward", values="pair_acc").sort("reward")
    sel = res.pivot(on="bucket", index="reward", values="sel_wer").sort("reward")
    with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=250):
        print("pairwise ranking accuracy (%)")
        print(wide)
        print("WER when the reward alone selects")
        print(sel)
        print(res.group_by("bucket").agg(pl.col("clips").first(), pl.col("pairs").first()))
    if out:
        res.write_csv(f"reports/{out}.csv")


if __name__ == "__main__":
    typer.run(main)
