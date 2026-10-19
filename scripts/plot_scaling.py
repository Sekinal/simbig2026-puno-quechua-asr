"""Figure: test-time scaling with perturbed audio views (hybrid selection vs pool oracle).

Reads ``reports/view_scaling_<pool>.csv`` (``scripts/view_scaling.py``) and writes
``reports/figures/view_scaling.{png,pdf,svg}``: small multiples per evaluation set, two series
(hybrid WER, oracle WER) against the number of perturbed views. Palette: the project's validated
two-slot categorical order (scripts/pool_overlap.py), checked with the dataviz validator.

    uv run python scripts/plot_scaling.py --csv reports/view_scaling_pool_dev_aug12.csv
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from lets_plot import (
    LetsPlot,
    aes,
    element_blank,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    ggsave,
    ggsize,
    labs,
    scale_color_manual,
    theme,
    theme_minimal2,
)

SETS = {"dev_unseen": "dev unseen sentences", "dev_spont": "dev spontaneous", "ood": "OOD radio"}
COLORS = {"hybrid selection": "#2a78d6", "oracle (best candidate)": "#eb6834"}


def main(
    csv: Path = typer.Option(Path("reports/view_scaling_pool_dev_aug12.csv")),
    out: Path = typer.Option(Path("reports/figures")),
) -> None:
    df = (
        pl.read_csv(csv)
        .filter(pl.col("set").is_in(list(SETS)))
        .unpivot(index=["set", "views"], on=["hybrid", "oracle"], variable_name="series")
        .with_columns(
            set=pl.col("set").replace_strict(SETS),
            series=pl.col("series").replace_strict(
                {"hybrid": "hybrid selection", "oracle": "oracle (best candidate)"}
            ),
        )
    )
    LetsPlot.setup_html(no_js=True)
    plot = (
        ggplot(df, aes("views", "value", color="series"))
        + geom_line(size=1.0)
        + geom_point(size=2.6)
        + facet_wrap("set", ncol=3, scales="free_y", order=0)
        + scale_color_manual(values=COLORS)
        + labs(x="perturbed views per adapter", y="WER (%)", color="")
        + theme_minimal2()
        + theme(panel_grid_minor=element_blank(), legend_position="top")
        + ggsize(660, 230)
    )
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        ggsave(plot, f"view_scaling.{ext}", path=str(out))


if __name__ == "__main__":
    typer.run(main)
