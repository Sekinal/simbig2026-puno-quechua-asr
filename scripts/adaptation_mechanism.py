"""What does adapting the speech encoder change? Coverage vs ranking, and which contrasts.

On a shared candidate pool (every candidate scored by every model; ``pool_ctrl_ed``):

1. **Coverage:** oracle WER of each model's own beam-8 list (+ its perturbed views): does the
   right transcript appear among the model's hypotheses at all?
2. **Ranking:** pairwise accuracy of each model's sequence log-likelihood on candidate pairs with a
   different number of word errors, and the WER of picking the candidate it scores highest.
3. **Which contrasts:** the ranking accuracy of decoder-only LoRA (v2) and enc+dec LoRA (ed) by
   the kind of difference between the two candidates: a single one-letter contrast (q/k, ejective,
   aspiration, vowel, other; ``scripts/error_types.py`` classes), a suffix, another word, an
   inserted/deleted word, or several words. 95% intervals of the difference resample clips
   (read speech: speakers x sentences, pigeonhole; spontaneous and OOD: clips, descriptive), as
   candidate pairs of one clip are dependent.

Clips <= 39 s (longer ones carry CTC candidates only). Writes ``reports/mechanism_*.csv``.

    uv run python scripts/adaptation_mechanism.py --cands pool_ctrl_ed
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from error_types import classify
from qxp_asr.metrics import official_normalize
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.stats import bootstrap_diffs, summarize

SETS = {
    "dev_unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
    "itest_unseen": (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"),
    "dev_spont": pl.col("role") == "dev_spont",
    "itest_spont": pl.col("role") == "itest_spont",
    "ood": pl.col("role") == "ood",
}
GROUPS = {
    "read": ("dev_unseen", "itest_unseen"),
    "spont": ("dev_spont", "itest_spont"),
    "ood": ("ood",),
}
SOURCES = {"zs": ["zs"], "v1": ["v1"], "v2": ["v2"], "ed": ["ed"]}
CONTRAST = {
    "q/k": "1-letter q/k",
    "ejective": "1-letter ejective",
    "aspiration": "1-letter aspiration",
    "vowel": "1-letter vowel",
    "other 1-letter": "1-letter other",
}


def pair_type(a: str, b: str) -> str:
    wa, wb = a.split(), b.split()
    ops = Levenshtein.editops(wa, wb)
    if len(ops) != 1:
        return "several words"
    op = ops[0]
    if op.tag != "replace":
        return "inserted/deleted word"
    kind = classify(wa[op.src_pos], wb[op.dest_pos])
    return CONTRAST.get(kind, "suffix" if kind == "suffix" else "other word")


def main(cands: str = typer.Option("pool_ctrl_ed"), out: str = typer.Option("mechanism")) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("text").is_not_null() & (pl.col("duration_s") <= 39)
    )
    pool = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet").with_columns(
        src=pl.col("sources").list.eval(pl.element().str.replace(r"(@aug|@smp)\d+$", "$1"))
    )
    scorers = [c for c in ("lp_zs", "lp_v1", "lp_v2", "lp_ed") if c in pool.columns]
    cov, rank = [], []
    # group -> pair type -> clip -> [pairs, v2 right, ed right]; clip -> (speaker, sentence)
    pairs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
    units: dict = {}
    for set_name, cond in SETS.items():
        d = man.filter(cond)
        refs = dict(d.select("clip_id", "text").iter_rows())
        units |= {
            c: (sp, se) for c, sp, se in d.select("clip_id", "speaker", "sentence_id").iter_rows()
        }
        by = pool.filter(pl.col("clip_id").is_in(list(refs))).partition_by("clip_id", as_dict=True)
        n_words = 0
        err_cov = defaultdict(int)
        err_sel = defaultdict(int)
        acc = defaultdict(list)
        for (cid,), g in by.items():
            r = official_normalize(refs[cid]).split()
            n_words += len(r)
            texts = [official_normalize(t) for t in g["text"]]
            e = np.array([Levenshtein.distance(r, t.split()) for t in texts])
            for name, keep in SOURCES.items():
                for suffix, extra in (("", []), ("+views", [f"{k}@aug" for k in keep])):
                    allowed = set(keep) | set(extra)
                    own = [
                        e[i] for i, s in enumerate(g["src"].to_list()) if allowed.intersection(s)
                    ]
                    err_cov[name + suffix] += min(own, default=len(r))
            better = e[:, None] < e[None, :]
            for sc in scorers:
                v = g[sc].fill_null(-1e9).to_numpy()
                err_sel[sc] += e[int(np.argmax(v))]
                if better.any():
                    acc[sc].append(((v[:, None] > v[None, :]) & better).sum() / better.sum())
            group = next(k for k, sets in GROUPS.items() if set_name in sets)
            if "lp_v2" in g.columns and "lp_ed" in g.columns:
                v2, ed = g["lp_v2"].fill_null(-1e9).to_list(), g["lp_ed"].fill_null(-1e9).to_list()
                for i in range(len(texts)):
                    for j in range(len(texts)):
                        if e[i] < e[j]:
                            for kind in (pair_type(texts[i], texts[j]), "all"):
                                s = pairs[group][kind][cid]
                                s[0] += 1
                                s[1] += v2[i] > v2[j]
                                s[2] += ed[i] > ed[j]
        cov += [
            {"set": set_name, "source": k, "oracle_wer": round(100 * v / n_words, 2)}
            for k, v in err_cov.items()
        ]
        rank += [
            {
                "set": set_name,
                "scorer": sc[3:],
                "pairwise_acc": round(100 * float(np.mean(acc[sc])), 2),
                "selection_wer": round(100 * err_sel[sc] / n_words, 2),
            }
            for sc in scorers
        ]
    pair_rows = []
    for grp, kinds in pairs.items():
        for kind, clips in sorted(kinds.items(), key=lambda x: -sum(v[0] for v in x[1].values())):
            cids = list(clips)
            n, a, b = (np.array([clips[c][k] for c in cids], dtype=float) for k in range(3))
            g1 = [units[c][0] for c in cids] if grp == "read" else cids
            g2 = [units[c][1] for c in cids] if grp == "read" else None
            lo, hi, p = summarize(bootstrap_diffs(b, a, n, g1, g2))
            pair_rows.append({
                "group": grp, "pair_type": kind, "pairs": int(n.sum()), "clips": len(cids),
                "v2_acc": round(100 * a.sum() / n.sum(), 2),
                "ed_acc": round(100 * b.sum() / n.sum(), 2),
                "delta": round(100 * (b.sum() - a.sum()) / n.sum(), 2),
                "lo": round(lo, 2), "hi": round(hi, 2), "p": round(p, 4),
            })  # fmt: skip
    for name, rows in (("coverage", cov), ("ranking", rank), ("pairs", pair_rows)):
        df = pl.DataFrame(rows)
        df.write_csv(f"reports/{out}_{name}.csv")
        with pl.Config(tbl_rows=80, tbl_width_chars=200):
            print(df)


if __name__ == "__main__":
    typer.run(main)
