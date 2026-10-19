"""Compare two systems with the paper's statistical designs (``scripts/paper_stats.py``).

A system is either a predictions table (``artifacts/preds/<name>.parquet``) or a weight-search
ablation run ``abl:<table>:<variant>`` (``artifacts/preds/ablation_<table>.parquet``, per-clip edit
counts averaged over its seeds). Prints the clip-level and design-based intervals per set.

    uv run python scripts/compare_runs.py --a abl:abl_pool_dev_aug_ilm:full \
        --b abl:hybrid_ablation:full
"""

from __future__ import annotations

import polars as pl
import typer

from paper_stats import Stats, ablation_systems

SETS = "dev_unseen,itest_unseen,dev_spont,itest_spont,ood"


def system(st: Stats, spec: str):
    if spec.startswith("abl:"):
        _, table, variant = spec.split(":", 2)
        return ablation_systems(st, table, variant)
    return spec


def main(
    a: str = typer.Option(...),
    b: str = typer.Option(...),
    sets: str = typer.Option(SETS),
    resamples: int = typer.Option(10_000),
) -> None:
    st = Stats(resamples)
    st.compare("compare", f"{a} - {b}", system(st, a), system(st, b), sets.split(","))
    cols = ["set", "wer_a", "wer_b", "diff", "clip_lo", "clip_hi", "design", "lo", "hi", "p"]
    with pl.Config(tbl_rows=20, tbl_width_chars=220, fmt_str_lengths=40):
        print(f"{a}  vs  {b}")
        print(pl.DataFrame(st.rows).select(cols))


if __name__ == "__main__":
    typer.run(main)
