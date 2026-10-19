"""Figure: achievable WER vs verifier ranking accuracy (``reports/verifier_curve.csv``).

Small multiples (one panel per evaluation set), two verifier families (blend toward the oracle,
noisy oracle; ``scripts/verifier_curve.py``), our verifier's operating point (the tuned hybrid)
marked and labelled. Colours: the validated two-slot palette.

    uv run python scripts/plot_verifier_curve.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from lets_plot import (
    LetsPlot,
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    ggsave,
    ggsize,
    labs,
    scale_color_manual,
    scale_x_continuous,
    theme,
    theme_minimal2,
)

SETS = {
    "dev_unseen": "dev unseen sentences",
    "itest_unseen": "itest unseen sentences",
    "dev_spont": "dev spontaneous",
    "ood": "OOD radio",
}


def main(
    csv: Path = typer.Option(Path("reports/verifier_curve.csv")),
    out: Path = typer.Option(Path("reports/figures")),
) -> None:
    df = pl.read_csv(csv).with_columns(
        rank=pl.col("set").replace_strict({k: i for i, k in enumerate(SETS)}),
        set=pl.col("set").replace_strict(SETS),
    )
    df = df.with_columns(
        family=pl.col("family").replace_strict(
            {"blend": "blend toward oracle", "noisy oracle": "noisy oracle"}
        )
    ).sort("rank", "family", "pair_acc")
    ours = df.filter((pl.col("family") == "blend toward oracle") & (pl.col("x") == 0))
    ours = ours.with_columns(label=pl.lit("ours"))
    LetsPlot.setup_html(no_js=True)
    plot = (
        ggplot(df, aes("pair_acc", "wer", color="family"))
        + geom_line(size=1.0)
        + geom_point(size=2.0)
        + geom_point(data=ours, color="#0b0b0b", size=4.0)
        + geom_text(aes(label="label"), data=ours, color="#0b0b0b", size=9, hjust=0, nudge_x=1.5)
        + scale_color_manual(values=["#2a78d6", "#eb6834"], name="")
        + facet_wrap("set", ncol=4, scales="free_y", order=0)
        + scale_x_continuous(limits=[55, 101])
        + labs(x="verifier pairwise ranking accuracy (%)", y="WER (%)")
        + theme_minimal2()
        + theme(
            panel_grid_minor=element_blank(),
            legend_position="top",
            text=element_text(size=15),
            strip_text=element_text(size=15),
        )
        + ggsize(1000, 235)
    )
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        ggsave(plot, f"verifier_curve.{ext}", path=str(out))
    svg = out / "verifier_curve.svg"  # end-of-file hook: lets-plot omits the final newline
    svg.write_text(svg.read_text().rstrip("\n") + "\n")


if __name__ == "__main__":
    typer.run(main)
