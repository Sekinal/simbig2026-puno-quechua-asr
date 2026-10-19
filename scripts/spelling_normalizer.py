"""Domain-conditional spelling normalisation of hypotheses (one-character orthographic variants).

Error analysis of the hybrid: 36-49 % of residual word errors are single-character substitutions,
dominated by orthographic conventions (h-aspiration, q/k, sh/s, ll/l, apostrophe) that differ
between the curated scripted references (Puno orthography: huq, ishkay, qulqi, ashwan) and the
spontaneous transcriptions (huk, iskay, ...); the models, trained on both, mix them. The domain of a
clip tells which convention its reference follows. From the TRAINING references of each domain
(role ``train``; scripted = validated prompts, spontaneous = gold transcriptions) we count word
frequencies; a hypothesis word ``w`` is replaced by its one-edit neighbour ``v`` if, in the clip's
domain, ``count(v) >= ratio * max(count(w), 1)`` and ``count(v) >= min_count`` (the most frequent
such neighbour wins). ``ratio`` / ``min_count`` are chosen on dev; itest and OOD only report.

    uv run python scripts/spelling_normalizer.py --pred hybrid_dev_aug
"""

from __future__ import annotations

from collections import Counter

import polars as pl
import typer

from qxp_asr.metrics import official_normalize, wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS

SETS = {
    "dev_unseen": ((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"), "scripted"),
    "dev_spont": (pl.col("role") == "dev_spont", "spont"),
    "itest_unseen": ((pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"), "scripted"),
    "itest_spont": (pl.col("role") == "itest_spont", "spont"),
    "ood": (pl.col("role") == "ood", "spont"),
}
GRID = [(r, m) for r in (2, 3, 5, 10, 20) for m in (2, 5, 10)]


# Orthographic convention edits seen in the error analysis (hyp <-> ref): aspiration after a stop
# or s, q/k, l/ll, the ejective apostrophe, and the vowel pairs u/o, i/e. Arbitrary one-character
# edits would turn correct but rare (agglutinated) words into frequent wrong ones.
SWAPS = [("q", "k"), ("u", "o"), ("i", "e"), ("ll", "l")]
ASPIRABLE = ("p", "t", "k", "q", "ch", "s")


def neighbours(w: str) -> set[str]:
    out = set()
    for a, b in SWAPS + [(y, x) for x, y in SWAPS]:
        i = w.find(a)
        while i >= 0:
            out.add(w[:i] + b + w[i + len(a) :])
            i = w.find(a, i + 1)
    for i in range(len(w)):
        if w[i] == "h" and any(w[:i].endswith(c) for c in ASPIRABLE):
            out.add(w[:i] + w[i + 1 :])  # drop aspiration
        if w[i] == "'":
            out.add(w[:i] + w[i + 1 :])  # drop ejective mark
        if any(w[: i + 1].endswith(c) for c in ASPIRABLE) and not w[i + 1 :].startswith("h"):
            out.add(w[: i + 1] + "h" + w[i + 1 :])  # add aspiration
        if any(w[: i + 1].endswith(c) for c in ("p", "t", "k", "q", "ch")):
            out.add(w[: i + 1] + "'" + w[i + 1 :])  # add ejective mark
    out.discard(w)
    return out


def counts(man: pl.DataFrame) -> dict[str, Counter]:
    train = man.filter(pl.col("role") == "train")
    scripted = train.filter((pl.col("corpus") == "scripted") & (pl.col("bucket") == "validated"))
    spont = train.filter(
        (pl.col("corpus") == "spontaneous") & pl.col("label").is_in(["gold", "pending"])
    )
    out = {}
    for name, df in (("scripted", scripted), ("spont", spont)):
        c: Counter = Counter()
        for t in df["text"].drop_nulls():
            c.update(official_normalize(t).split())
        out[name] = c
    return out


def normalise(text: str, c: Counter, ratio: float, min_count: int, cache: dict) -> str:
    words = []
    for w in official_normalize(text or "").split():
        key = (id(c), w, ratio, min_count)
        if key not in cache:
            base = max(c[w], 1)
            best = max(
                (v for v in neighbours(w) if c[v] >= min_count and c[v] >= ratio * base),
                key=lambda v: c[v],
                default=w,
            )
            cache[key] = best
        words.append(cache[key])
    return " ".join(words)


def main(pred: str = typer.Option(...), out: str = typer.Option("")) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    c = counts(man)
    preds = pl.read_parquet(ARTIFACTS / "preds" / f"{pred}.parquet").filter(
        pl.col("text").is_not_null()
    )
    cache: dict = {}
    rows = []
    for name, (cond, domain) in SETS.items():
        g = preds.filter(cond)
        refs, hyps = g["text"].to_list(), g["hyp"].to_list()
        base = wer_cer(refs, hyps)["wer"]
        for ratio, min_count in GRID:
            new = [normalise(h, c[domain], ratio, min_count, cache) for h in hyps]
            rows.append(
                {
                    "set": name,
                    "ratio": ratio,
                    "min_count": min_count,
                    "base": round(base, 2),
                    "norm": round(wer_cer(refs, new)["wer"], 2),
                }
            )
    res = pl.DataFrame(rows).with_columns(delta=pl.col("norm") - pl.col("base"))
    dev = res.filter(pl.col("set").str.starts_with("dev"))
    best = dev.group_by("ratio", "min_count").agg(pl.col("delta").mean()).sort("delta").row(0)
    print(f"chosen on dev: ratio={best[0]}, min_count={best[1]} (mean dev change {best[2]:+.2f})")
    with pl.Config(tbl_rows=20):
        print(res.filter((pl.col("ratio") == best[0]) & (pl.col("min_count") == best[1])))
    if out:  # write the normalised predictions for the chosen setting
        normed = preds.with_columns(
            hyp=pl.struct("hyp", "role", "test_type").map_elements(
                lambda r: normalise(
                    r["hyp"],
                    c[
                        "spont"
                        if r["role"] in ("dev_spont", "itest_spont", "ood")
                        or r["test_type"] == "spontaneous"
                        else "scripted"
                    ],
                    best[0],
                    best[1],
                    cache,
                ),
                return_dtype=pl.String,
            )
        )
        normed.write_parquet(ARTIFACTS / "preds" / f"{out}.parquet")


if __name__ == "__main__":
    typer.run(main)
