"""Does the hidden test reuse the Common Voice scripted sentence pool?

For each hypothesis we compute the normalized character edit distance to the closest sentence of
the *training* pool (role ``train``; held-out "unseen" sentences are excluded from the pool).
Calibration: dev clips whose sentence is in the pool vs dev clips with held-out sentences.
If the test distribution looks like the in-pool calibration, closed-set LM biasing pays off.
"""

from __future__ import annotations

import polars as pl
import typer
from lets_plot import (
    LetsPlot,
    aes,
    element_blank,
    geom_density,
    ggplot,
    ggsave,
    ggsize,
    labs,
    scale_fill_manual,
    theme,
    theme_minimal2,
)
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from qxp_asr.paths import ARTIFACTS, MANIFESTS, REPORTS
from qxp_asr.text import normalize


def main(preds: str = typer.Option("zs_llm7b")) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    pool = (
        man.filter((pl.col("role") == "train") & (pl.col("corpus") == "scripted"))["text"]
        .unique()
        .to_list()
    )
    pred = pl.read_parquet(ARTIFACTS / "preds" / f"{preds}.parquet").filter(
        (pl.col("role") == "dev_scripted") | (pl.col("test_type") == "scripted")
    )
    hyps = [normalize(h) for h in pred["hyp"].to_list()]
    best = [process.extractOne(h, pool, scorer=Levenshtein.normalized_distance) for h in hyps]
    df = pred.with_columns(
        dist=pl.Series([b[1] for b in best]),
        group=pl.when(pl.col("role") == "test")
        .then(pl.lit("test_scripted"))
        .when(pl.col("unseen_sentence"))
        .then(pl.lit("dev_unseen (not in pool)"))
        .otherwise(pl.lit("dev_seen (in pool)")),
    )
    summary = (
        df.group_by("group")
        .agg(
            n=pl.len(),
            median=pl.col("dist").median().round(3),
            p75=pl.col("dist").quantile(0.75).round(3),
            p90=pl.col("dist").quantile(0.90).round(3),
            frac_lt_0_15=(pl.col("dist") < 0.15).mean().round(3),
            frac_lt_0_25=(pl.col("dist") < 0.25).mean().round(3),
        )
        .sort("group")
    )
    with pl.Config(tbl_width_chars=200):
        print(summary)
    LetsPlot.setup_html(no_js=True)
    colors = {  # validated categorical palette (dataviz reference), fixed order
        "dev_seen (in pool)": "#2a78d6",
        "dev_unseen (not in pool)": "#eb6834",
        "test_scripted": "#1baf7a",
    }
    plot = (
        ggplot(df.select("dist", "group"), aes("dist", fill="group"))
        + geom_density(alpha=0.55, color="white", size=0.3)
        + scale_fill_manual(values=colors)
        + labs(
            title="Distance of scripted hypotheses to the closest Common Voice sentence",
            x="normalized character edit distance",
            y="density",
            fill="",
        )
        + theme_minimal2()
        + theme(panel_grid_minor=element_blank(), legend_position="top")
        + ggsize(760, 360)
    )
    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        ggsave(plot, f"pool_distance.{ext}", path=str(out))


if __name__ == "__main__":
    typer.run(main)
