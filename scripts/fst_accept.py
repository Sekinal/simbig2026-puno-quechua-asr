"""Word acceptance by a Southern Quechua morphological FST (a reference-free linguistic verifier).

Uses the unified Southern Quechua spell-checking transducer of the squoia project (Rios, 2015;
https://github.com/a-rios/squoia, ``morphology/spellcheck_plugin/src/oxt/lib/foma/qhichwa.fst``,
gunzipped to ``artifacts/fst/qhichwa.bin``). A word is accepted if the transducer returns any
analysis. Every distinct word of the given candidate tables and of all gold transcripts is looked
up once; ``scripts/reward_bench.py`` turns this into a per-candidate reward.

hfst wheels need Python 3.11, so this runs in its own small environment (``envs/fst``):
    uv run python /workspace/simbig/repo/scripts/fst_accept.py --cands pool_dev_tc
"""

from __future__ import annotations

from pathlib import Path

import hfst
import polars as pl
import typer

ROOT = Path("/workspace/simbig")


def main(
    cands: list[str] = typer.Option(..., help="candidate tables in artifacts/cands"),
    fst: Path = typer.Option(ROOT / "artifacts" / "fst" / "qhichwa.bin"),
    out: Path = typer.Option(ROOT / "artifacts" / "fst" / "accept.parquet"),
) -> None:
    texts = [pl.read_parquet(ROOT / "artifacts" / "cands" / f"{c}.parquet")["text"] for c in cands]
    gold = pl.read_parquet(ROOT / "data" / "manifests" / "all.parquet")["text"].drop_nulls()
    words = (
        pl.concat([*texts, gold.str.to_lowercase()])
        .str.split(" ")
        .explode()
        .str.strip_chars('.,;:!?¿¡"()')
        .unique()
        .drop_nulls()
    )
    words = [w for w in words.to_list() if w]
    t = hfst.HfstInputStream(str(fst)).read()
    acc = [len(t.lookup(w)) > 0 for w in words]
    pl.DataFrame({"word": words, "accepted": acc}).write_parquet(out)
    print(f"{len(words)} words, {100 * sum(acc) / len(words):.1f} % accepted -> {out}")


if __name__ == "__main__":
    typer.run(main)
