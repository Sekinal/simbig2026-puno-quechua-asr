"""Which kinds of word errors does a system make? Phonetic-contrast breakdown of WER.

Hypothesis and reference are aligned word by word (Levenshtein edit operations). Substitutions are
classified by the Puno Quechua orthographic contrast that separates the two words, checked in this
order: uvular/velar ``q``/``k``; ejective apostrophe; aspiration ``h`` after a stop; vowel lowering
``i``/``e``, ``u``/``o`` (Spanish-influenced spellings); any other one-letter difference; more than
one letter, split into "suffix" (same stem: a common prefix of at least 3 letters and half the
shorter word, i.e. a different ending in this agglutinative language) and "other word".
Insertions and deletions are counted separately. Counts are per 100 reference words, so the rows
of a system sum to its WER.

    uv run python scripts/error_types.py --pred llm7b_lora_v2 --pred llm7b_lora_encdec
"""

from __future__ import annotations

import re

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
CLASSES = [
    "q/k",
    "ejective",
    "aspiration",
    "vowel",
    "other 1-letter",
    "suffix",
    "other word",
    "ins",
    "del",
]


def _strip(w: str, kind: str) -> str:
    if kind == "q/k":
        return w.replace("q", "k")
    if kind == "ejective":
        return w.replace("'", "")
    if kind == "aspiration":
        return re.sub(r"(ch|[ptkq])h", r"\1", w)
    return w.replace("e", "i").replace("o", "u")  # vowel


def _common_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def classify(ref: str, hyp: str) -> str:
    for kind in ("q/k", "ejective", "aspiration", "vowel"):
        if _strip(ref, kind) == _strip(hyp, kind):
            return kind
    if Levenshtein.distance(ref, hyp) == 1:
        return "other 1-letter"
    stem = _common_prefix(ref, hyp)
    if stem >= 3 and stem >= 0.5 * min(len(ref), len(hyp)):  # same stem, different ending
        return "suffix"
    return "other word"


def errors(ref: str, hyp: str) -> list[str]:
    r, h = official_normalize(ref).split(), official_normalize(hyp or "").split()
    out = []
    for op in Levenshtein.editops(r, h):
        if op.tag == "replace":
            out.append(classify(r[op.src_pos], h[op.dest_pos]))
        else:
            out.append("ins" if op.tag == "insert" else "del")
    return out


def main(pred: list[str] = typer.Option(...), out: str = typer.Option("error_types")) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("text").is_not_null())
    rows = []
    for name in pred:
        hyps = dict(
            pl.read_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
            .select("clip_id", "hyp")
            .iter_rows()
        )
        for set_name, cond in SETS.items():
            d = man.filter(cond & pl.col("clip_id").is_in(list(hyps)))
            counts = dict.fromkeys(CLASSES, 0)
            n_words = 0
            for cid, ref in d.select("clip_id", "text").iter_rows():
                n_words += len(official_normalize(ref).split())
                for e in errors(ref, hyps[cid]):
                    counts[e] += 1
            rows.append(
                {"system": name, "set": set_name}
                | {k: round(100 * v / n_words, 2) for k, v in counts.items()}
                | {"wer": round(100 * sum(counts.values()) / n_words, 2)}
            )
    df = pl.DataFrame(rows)
    df.write_csv(f"reports/{out}.csv")
    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=220):
        print(df)


if __name__ == "__main__":
    typer.run(main)
