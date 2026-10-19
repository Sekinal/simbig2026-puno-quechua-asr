"""Word-level error analysis: how much WER is orthographic-convention mismatch?

For every reference/hypothesis pair we align words (Levenshtein edit operations) and classify each
substitution as *orthographic* when both words map to the same canonical Puno spelling under
``qxp_asr.respell`` (e.g. ``qullqi``/``qulqi``, ``huk``/``huq``, ``aswan``/``ashwan``) — i.e. the
system heard the word correctly but used the other spelling convention. The spontaneous gold
transcripts and the organizers' curated scripted sentences follow different conventions.
"""

from __future__ import annotations

from collections import Counter

import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from qxp_asr.paths import ARTIFACTS
from qxp_asr.respell import respell


def analyse(refs: list[str], hyps: list[str]) -> dict:
    n_ref = subs = ins = dels = ortho = 0
    pairs: Counter[tuple[str, str]] = Counter()
    for ref, hyp in zip(refs, hyps, strict=True):
        r, h = ref.split(), hyp.split()
        n_ref += len(r)
        for op in Levenshtein.editops(r, h):
            if op.tag == "replace":
                subs += 1
                a, b = r[op.src_pos], h[op.dest_pos]
                pairs[(a, b)] += 1
                if respell(a) == respell(b):
                    ortho += 1
            elif op.tag == "insert":
                ins += 1
            else:
                dels += 1
    return {
        "wer": 100 * (subs + ins + dels) / n_ref,
        "sub": 100 * subs / n_ref,
        "ins": 100 * ins / n_ref,
        "del": 100 * dels / n_ref,
        "ortho_sub": 100 * ortho / n_ref,
        "top_pairs": pairs.most_common(15),
    }


def main(
    preds: list[str] = typer.Option(...),
    role: str = typer.Option("dev_spont"),
    col: str = typer.Option("hyp"),
) -> None:
    for name in preds:
        df = pl.read_parquet(ARTIFACTS / "preds" / f"{name}.parquet").filter(
            (pl.col("role") == role) & pl.col("text").is_not_null()
        )
        res = analyse(df["text"].to_list(), df[col].fill_null("").to_list())
        top = res.pop("top_pairs")
        print(f"== {name} [{role}] " + " ".join(f"{k}={v:.2f}" for k, v in res.items()))
        print("   top substitutions (ref -> hyp):", ", ".join(f"{a}->{b}:{n}" for (a, b), n in top))


if __name__ == "__main__":
    typer.run(main)
