"""Named training sources built from the manifest (and optional pseudo-label tables).

``mode="dev"``  trains only on role ``train`` (dev / itest / held-out sentences stay unseen).
``mode="org"``  adds held-out-sentence clips of training speakers (= organizer training condition).
``mode="final"`` additionally uses every holdout and itest clip (dev kept for monitoring only).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from qxp_asr.text import normalize

MIN_S, MAX_S = 0.5, 35.0

SOURCES: dict[str, pl.Expr] = {
    "scripted": (pl.col("corpus") == "scripted") & (pl.col("bucket") == "validated"),
    "scripted_other": (pl.col("corpus") == "scripted") & (pl.col("bucket") != "validated"),
    "spont_gold": (pl.col("corpus") == "spontaneous") & pl.col("label").is_in(["gold", "pending"]),
    "spont_silver": (pl.col("corpus") == "spontaneous") & (pl.col("label") == "silver"),
}


def roles_for_mode(mode: str) -> list[str]:
    if mode == "dev":
        return ["train"]
    if mode == "org":  # organizer-equivalent: all sentences, dev/itest speakers unseen
        return ["train", "holdout_sent"]
    if mode == "final":
        return ["train", "holdout_sent", "holdout", "itest_scripted", "itest_spont"]
    msg = f"unknown mode {mode!r}"
    raise ValueError(msg)


def training_rows(
    man: pl.DataFrame,
    sources: list[str],
    mode: str = "dev",
    pseudo: dict[str, Path] | None = None,
) -> pl.DataFrame:
    """Return rows (clip_id, wav, duration_s, target, source) for the requested sources.

    ``pseudo`` maps a source name to a parquet with columns (clip_id, target); those labels
    replace whatever the manifest has for the listed clips. A table that also has ``wav`` and
    ``duration_s`` brings its own audio (e.g. synthetic speech, ``scripts/tts_synth.py``) and is
    used as is.
    """
    base = man.filter(pl.col("role").is_in(roles_for_mode(mode)))
    frames = []
    for name in sources:
        if name in SOURCES:
            target = pl.col("silver") if name == "spont_silver" else pl.col("text")
            frames.append(
                base.filter(SOURCES[name]).with_columns(target=target, source=pl.lit(name))
            )
        elif pseudo and name in pseudo:
            table = pl.read_parquet(pseudo[name])
            if {"wav", "duration_s"} <= set(table.columns):  # own audio, not in the manifest
                frames.append(table.with_columns(source=pl.lit(name)))
            else:
                labels = table.select("clip_id", "target")
                frames.append(base.join(labels, on="clip_id").with_columns(source=pl.lit(name)))
        else:
            msg = f"unknown source {name!r}"
            raise KeyError(msg)
    rows = pl.concat([f.select("clip_id", "wav", "duration_s", "target", "source") for f in frames])
    return rows.with_columns(
        target=pl.col("target").map_elements(normalize, return_dtype=pl.String)
    ).filter(
        pl.col("duration_s").is_between(MIN_S, MAX_S)
        & (pl.col("target").str.len_chars() > 0)
        # CTC needs enough frames: 20 ms per frame, so at most 50 chars per second.
        & (pl.col("target").str.len_chars() < pl.col("duration_s") * 40)
    )


def pseudo_pool(man: pl.DataFrame, mode: str = "dev", max_s: float = 39.0) -> pl.DataFrame:
    """Unlabelled spontaneous clips eligible for pseudo-labelling (silver, untranscribed, segments).

    Only speakers allowed for training in ``mode`` are included, so dev/itest speakers stay unseen.
    """
    return man.filter(
        pl.col("role").is_in(roles_for_mode(mode))
        & (pl.col("corpus") == "spontaneous")
        & ~pl.col("label").is_in(["gold", "pending"])
        & pl.col("duration_s").is_between(MIN_S, max_s)
    )
