"""Build ``data/manifests/all.parquet`` (see ``qxp_asr.manifest`` for the schema)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import soundfile as sf

from qxp_asr.manifest import build
from qxp_asr.paths import MANIFESTS
from qxp_asr.splits import assign_roles


def _duration(path: str) -> float | None:
    try:
        return sf.info(path).duration
    except RuntimeError:
        return None


def main() -> None:
    df = build()
    df = df.with_columns(
        wav_exists=pl.col("wav").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean),
    ).with_columns(
        duration_s=pl.when(pl.col("duration_s").is_null() & pl.col("wav_exists"))
        .then(pl.col("wav").map_elements(_duration, return_dtype=pl.Float64))
        .otherwise(pl.col("duration_s")),
    )
    df = assign_roles(df)
    assert df["clip_id"].is_unique().all(), "duplicate clip ids"
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS / "all.parquet"
    df.write_parquet(out)
    print(
        df.group_by("corpus", "bucket", "label")
        .agg(
            n=pl.len(),
            hours=(pl.col("duration_s").sum() / 3600).round(2),
            missing_wav=(~pl.col("wav_exists")).sum(),
        )
        .sort("corpus", "bucket", "label")
    )
    print(
        df.group_by("role", "corpus", "unseen_sentence")
        .agg(
            n=pl.len(),
            hours=(pl.col("duration_s").sum() / 3600).round(2),
            speakers=pl.col("speaker").n_unique(),
            labelled=pl.col("text").is_not_null().sum(),
        )
        .sort("role", "corpus", "unseen_sentence")
    )
    print(f"wrote {out} ({df.height} rows)")


if __name__ == "__main__":
    main()
