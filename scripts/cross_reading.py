"""Other readings of the same sentence as an orthogonal signal (transductive, scripted speech).

Every clip has so far been decoded alone. In Common Voice, a sentence is read by several speakers,
and 72 % of errors on twice-read held-out sentences hit a word that the other readings got right.
Here a clip's *neighbours* are the other clips of the same evaluation set whose hybrid 1-best is
within a normalized character distance ``tau`` of its own (no labels used); each candidate gets a
cross-reading penalty, the mean word edit distance to the neighbours' 1-best hypotheses, and

    score' = hybrid score - lam * penalty

``lam`` and ``tau`` are tuned on dev unseen and applied to itest unseen. An oracle variant groups
clips by their true sentence id (upper bound). Writes a predictions table for paired tests and
reports how many test clips would have neighbours.

    uv run python scripts/cross_reading.py --cands pool_ctrl_ed_scoreonly \
        --weights hybrid_dev_ed.weights.json --base hybrid_dev_ed
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

SETS = {
    "dev_unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
    "itest_unseen": (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"),
}


def neighbours(hyps: dict[str, str], tau: float) -> dict[str, list[str]]:
    ids = list(hyps)
    norm = {c: official_normalize(hyps[c]) for c in ids}
    out = {c: [] for c in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if Levenshtein.normalized_distance(norm[a], norm[b]) <= tau:
                out[a].append(b)
                out[b].append(a)
    return out


def rescore(clips, scores, nb: dict[str, list[str]], hyps: dict[str, str], lam: float) -> dict:
    out = {}
    for cid, texts in clips.items():
        s = scores[cid]
        if not nb.get(cid) or lam == 0:
            out[cid] = texts[int(np.argmax(s))]
            continue
        others = [official_normalize(hyps[n]).split() for n in nb[cid]]
        pen = np.array(
            [np.mean([Levenshtein.distance(t.split(), o) for o in others]) for t in texts]
        )
        out[cid] = texts[int(np.argmax(s - lam * pen))]
    return out


def main(
    cands: str = typer.Option("pool_ctrl_ed_scoreonly"),
    weights: str = typer.Option("hybrid_dev_ed.weights.json"),
    base: str = typer.Option("hybrid_dev_ed", help="predictions table to extend"),
    post: str = typer.Option("ctc1b_v3_soup_wise0.85"),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("indomain"),
    test_pred: str = typer.Option(
        "hybrid_final_ed", help="final predictions: test neighbour count"
    ),
    workers: int = typer.Option(8),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    w_all = json.loads((ARTIFACTS / "cands" / weights).read_text())["scripted"]
    names = ["chr", "len", *lp_cols, "naug"]
    w = np.array([w_all[k] for k in names])
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("text").is_not_null())
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    data = {}
    for name, cond in SETS.items():
        d = man.filter(cond)
        feats = build_features(table, d["clip_id"].to_list(), post, char_lm, workers, False, True)
        clips = {c: feats[c][0] for c in feats}
        scores = {c: feats[c][1][:, 0] + feats[c][1][:, 1:] @ w for c in feats}
        hyps = {c: clips[c][int(np.argmax(scores[c]))] for c in clips}
        refs = dict(d.select("clip_id", "text").iter_rows())
        sent = dict(d.select("clip_id", "sentence_id").iter_rows())
        oracle_nb = {c: [o for o in clips if o != c and sent[o] == sent[c]] for c in clips}
        data[name] = (clips, scores, hyps, refs, oracle_nb)

    def wer(name: str, picks: dict) -> float:
        refs = data[name][3]
        ids = list(picks)
        return wer_cer([refs[c] for c in ids], [picks[c] for c in ids])["wer"]

    rows, best = [], None
    for tau in (0.15, 0.25, 0.35, 0.45):
        nb = {n: neighbours(data[n][2], tau) for n in SETS}
        for lam in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            r = {"grouping": f"hyp tau={tau}", "lam": lam}
            for n in SETS:
                clips, scores, hyps, _, _ = data[n]
                r[n] = round(wer(n, rescore(clips, scores, nb[n], hyps, lam)), 2)
                r[f"{n}_with_nb"] = sum(1 for c in clips if nb[n][c])
            rows.append(r)
            if best is None or r["dev_unseen"] < best[0]["dev_unseen"]:
                best = (r, tau, lam)
    for lam in (0.5, 1.0, 2.0, 4.0, 8.0):  # oracle grouping by true sentence id (upper bound)
        r = {"grouping": "oracle sentence id", "lam": lam}
        for n in SETS:
            clips, scores, hyps, _, onb = data[n]
            r[n] = round(wer(n, rescore(clips, scores, onb, hyps, lam)), 2)
            r[f"{n}_with_nb"] = sum(1 for c in clips if onb[c])
        rows.append(r)
    df = pl.DataFrame(rows)
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(df)
    r, tau, lam = best
    print(f"selected on dev unseen: tau={tau} lam={lam} -> itest unseen {r['itest_unseen']}")
    # predictions with the dev-selected setting, for paired tests (other clips: base system)
    preds = pl.read_parquet(ARTIFACTS / "preds" / f"{base}.parquet")
    new = {}
    for n in SETS:
        clips, scores, hyps, _, _ = data[n]
        new |= rescore(clips, scores, neighbours(hyps, tau), hyps, lam)
    preds = preds.with_columns(
        hyp=pl.col("clip_id").replace_strict(new, default=pl.col("hyp"), return_dtype=pl.String)
    )
    preds.write_parquet(ARTIFACTS / "preds" / f"{base}_xread.parquet")
    # how many scripted test clips would have neighbours at this tau?
    tp = pl.read_parquet(ARTIFACTS / "preds" / f"{test_pred}.parquet")
    full = pl.read_parquet(MANIFESTS / "all.parquet")  # test clips have no text: full manifest
    test_scr = full.filter((pl.col("role") == "test") & (pl.col("test_type") == "scripted"))
    th = dict(
        tp.filter(pl.col("clip_id").is_in(test_scr["clip_id"].implode()))
        .select("clip_id", "hyp")
        .iter_rows()
    )
    tnb = neighbours(th, tau)
    with_nb = sum(1 for c in th if tnb[c])
    print(f"test scripted: {with_nb}/{len(th)} clips have a neighbour at tau={tau}")


if __name__ == "__main__":
    typer.run(main)
