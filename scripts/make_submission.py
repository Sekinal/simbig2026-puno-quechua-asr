"""Write the Task-1 submission file ``qxp.tsv`` from prediction tables.

Format (test README): no header, one ``<file_name>\\t<transcription>`` line per test clip, in the
order of the organizers' template ``qxp.tsv``. Hypotheses are normalized to the curated alphabet;
an empty hypothesis falls back to the next prediction table given (``--pred`` order = priority).

The output stays on the GPU box (``artifacts/submission``); it contains only our system output.
"""

from __future__ import annotations

import polars as pl
import typer

from qxp_asr.metrics import official_normalize
from qxp_asr.paths import ARTIFACTS, TEST_DIR
from qxp_asr.text import normalize


def main(
    pred: list[str] = typer.Option(..., help="prediction parquet names, highest priority first"),
    name: str = typer.Option("qxp"),
) -> None:
    template = pl.read_csv(
        TEST_DIR / "qxp.tsv", separator="\t", has_header=False, quote_char=None,
        new_columns=["file_name", "placeholder"],
    )  # fmt: skip
    out = template.select("file_name", clip_id=pl.col("file_name").str.replace(r"\.wav$", ""))
    out = out.with_columns(transcription=pl.lit(None, pl.String))
    for p in pred:
        tab = pl.read_parquet(ARTIFACTS / "preds" / f"{p}.parquet").select(
            "clip_id", h=pl.col("hyp").map_elements(normalize, return_dtype=pl.String)
        )
        out = (
            out.join(tab, on="clip_id", how="left")
            .with_columns(
                transcription=pl.when(
                    pl.col("transcription").is_null() | (pl.col("transcription") == "")
                )
                .then(pl.col("h"))
                .otherwise(pl.col("transcription"))
            )
            .drop("h")
        )
    missing = out.filter(pl.col("transcription").is_null() | (pl.col("transcription") == ""))
    if missing.height:
        print(f"WARNING: {missing.height} clips without a hypothesis (written as empty)")
    out = out.with_columns(pl.col("transcription").fill_null(""))
    # idempotence under the official scorer normalization: nothing we emit should change
    changed = sum(t != official_normalize(t) for t in out["transcription"].to_list())
    assert changed == 0, f"{changed} transcriptions change under official normalization"
    assert out.height == template.height == 659, out.height
    dst = ARTIFACTS / "submission" / f"{name}.tsv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{f}\t{t}" for f, t in out.select("file_name", "transcription").iter_rows()]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dst} ({out.height} lines); systems in priority order: {pred}")


if __name__ == "__main__":
    typer.run(main)
