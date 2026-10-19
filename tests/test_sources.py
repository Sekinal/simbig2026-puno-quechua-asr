"""Training-source assembly (qxp_asr.sources)."""

import polars as pl

from qxp_asr.sources import training_rows


def _manifest() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "clip_id": ["a", "b"],
            "wav": ["a.wav", "b.wav"],
            "duration_s": [3.0, 4.0],
            "text": ["wasi", None],
            "silver": [None, "runa"],
            "corpus": ["scripted", "spontaneous"],
            "bucket": ["validated", "other"],
            "label": ["gold", "none"],
            "role": ["train", "train"],
        }
    )


def test_pseudo_labels_join_the_manifest(tmp_path):
    p = tmp_path / "pseudo.parquet"
    pl.DataFrame({"clip_id": ["b"], "target": ["Runa"]}).write_parquet(p)
    rows = training_rows(_manifest(), ["scripted", "pl"], "dev", {"pl": p})
    assert rows.sort("clip_id")["target"].to_list() == ["wasi", "runa"]


def test_table_with_own_audio_is_used_as_is(tmp_path):
    p = tmp_path / "tts.parquet"
    pl.DataFrame(
        {"clip_id": ["tts_0"], "wav": ["tts_0.wav"], "duration_s": [2.5], "target": ["Allillanmi"]}
    ).write_parquet(p)
    rows = training_rows(_manifest(), ["tts"], "dev", {"tts": p})
    assert rows.select("clip_id", "wav", "target", "source").row(0) == (
        "tts_0",
        "tts_0.wav",
        "allillanmi",
        "tts",
    )
