"""Combine several systems' 1-best outputs (word-level ROVER or MBR medoid) and score them.

ROVER (Fiscus, 1997) / MBR system selection (Goel & Byrne, 2000) over heterogeneous systems
(CTC + n-gram LM vs LLM-ASR) — complementary error patterns are what make combination pay off
(docs/LITERATURE.md §g.3).

    uv run python scripts/combine_systems.py --sys llm7b_lora_v1:1.5 \
        --sys ctc1b_v2__rescore_all_all:1 --sys zs_llm7b_all:0.7 --name rover_v1
"""

from __future__ import annotations

import polars as pl
import typer

from qxp_asr.combine import medoid, rover
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.text import normalize

ROLES = ["dev_scripted", "dev_spont", "itest_scripted", "itest_spont", "ood", "test"]


def load(spec: str) -> tuple[str, float, pl.DataFrame]:
    name, _, w = spec.partition(":")
    df = pl.read_parquet(ARTIFACTS / "preds" / f"{name}.parquet").select(
        "clip_id", pl.col("hyp").fill_null("").map_elements(normalize, return_dtype=pl.String)
    )
    return name, float(w or 1.0), df.rename({"hyp": name})


def main(
    sys: list[str] = typer.Option(..., help="pred name[:weight]; first = ROVER pivot"),
    name: str = typer.Option(...),
    method: str = typer.Option("rover", help="rover | medoid"),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("role").is_in(ROLES))
    df = man.select("clip_id", "role", "unseen_sentence", "text", "test_type", "duration_s")
    systems = [load(s) for s in sys]
    for _, _, tab in systems:
        df = df.join(tab, on="clip_id", how="inner")
    names, weights = [s for s, _, _ in systems], [w for _, w, _ in systems]
    fn = rover if method == "rover" else medoid
    combined = [fn([r[n] for n in names], weights) for r in df.iter_rows(named=True)]
    df = df.with_columns(hyp=pl.Series(combined))
    df.write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
    lab = df.filter(pl.col("text").is_not_null())
    rows = [
        {"role": role, "unseen": unseen, "system": col}
        | wer_cer(g["text"].to_list(), g[col].to_list())
        for (role, unseen), g in lab.group_by("role", "unseen_sentence")
        for col in [*names, "hyp"]
    ]
    table = pl.DataFrame(rows).sort("role", "unseen", "system")
    with pl.Config(tbl_rows=80, tbl_width_chars=200, fmt_str_lengths=40):
        print(table.with_columns(pl.col("wer", "cer").round(2)))


if __name__ == "__main__":
    typer.run(main)
