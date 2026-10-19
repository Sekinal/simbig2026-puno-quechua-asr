"""Build one tidy clip-level manifest (polars) covering every qxp audio clip we have.

Columns
-------
clip_id       mp3/wav stem, unique key
corpus        "scripted" | "spontaneous" | "test" | "ood" (organizer radio drama, eval only)
speaker       hashed CV client_id (null for the shared-task test set)
duration_s    clip duration in seconds
wav           path of the 16 kHz mono WAV
text          best available target text, normalized (null when unlabelled)
label         provenance of ``text``: curated | gold | pending | silver | prompt | none
silver        organizer omniASR-LLM-7B silver transcript (spontaneous only), normalized
bucket        CV bucket: validated | invalidated | other (scripted); validated | pending |
              untranscribed (spontaneous); test
cv_split      official CV split (train/dev/test) or null
org_split     organizer ``splits_cv26`` split (train/dev/test) or null
sentence_id   scripted sentence id; prompt_id for spontaneous
prompt        spontaneous question prompt (normalized)
gender, age   self-declared demographics
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from qxp_asr.paths import (
    AUDIO,
    AUDIO_OOD,
    EXTERNAL,
    MANIFESTS,
    SCRIPTED_DIR,
    SPONTANEOUS_DIR,
    TEST_DIR,
)
from qxp_asr.text import normalize


def read_tsv(path: Path) -> pl.DataFrame:
    """Read a Common Voice style TSV (no quoting, everything as string)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def _norm(col: str) -> pl.Expr:
    return pl.col(col).map_elements(normalize, return_dtype=pl.String)


def _null_if_empty(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.str.len_chars() > 0).then(expr).otherwise(None)


def _org_split(corpus: str, key: str) -> pl.DataFrame:
    frames = [
        read_tsv(EXTERNAL / "splits_cv26" / corpus / f"{split}.tsv").with_columns(
            org_split=pl.lit(split)
        )
        for split in ("train", "dev", "test")
    ]
    return pl.concat(frames, how="diagonal")


def scripted() -> pl.DataFrame:
    curated = read_tsv(EXTERNAL / "shared-task" / "validated_sentences_curated.tsv")
    buckets = [
        read_tsv(SCRIPTED_DIR / f"{b}.tsv").with_columns(bucket=pl.lit(b))
        for b in ("validated", "invalidated", "other")
    ]
    clips = pl.concat(buckets, how="diagonal")
    cv_split = pl.concat(
        [
            read_tsv(SCRIPTED_DIR / f"{s}.tsv").select("path").with_columns(cv_split=pl.lit(s))
            for s in ("train", "dev", "test")
        ]
    )
    durations = read_tsv(SCRIPTED_DIR / "clip_durations.tsv").rename(
        {"clip": "path", "duration[ms]": "duration_ms"}
    )
    org = _org_split("scripted", "path").select("path", "org_split")
    return (
        clips.join(curated, on="sentence_id", how="left")
        .join(cv_split, on="path", how="left")
        .join(durations, on="path", how="left")
        .join(org, on="path", how="left")
        .with_columns(
            text=pl.coalesce(pl.col("sentence_curated"), _norm("sentence")),
            label=pl.when(pl.col("sentence_curated").is_not_null())
            .then(pl.lit("curated"))
            .otherwise(pl.lit("prompt")),
        )
        .select(
            clip_id=pl.col("path").str.replace(r"\.mp3$", ""),
            corpus=pl.lit("scripted"),
            speaker=pl.col("client_id"),
            duration_s=pl.col("duration_ms").cast(pl.Float64) / 1000,
            text="text",
            label="label",
            silver=pl.lit(None, pl.String),
            bucket="bucket",
            cv_split="cv_split",
            org_split="org_split",
            sentence_id="sentence_id",
            prompt=pl.lit(None, pl.String),
            gender="gender",
            age="age",
            up_votes=pl.col("up_votes").cast(pl.Int32),
            down_votes=pl.col("down_votes").cast(pl.Int32),
        )
    )


def spontaneous() -> pl.DataFrame:
    corpus = read_tsv(SPONTANEOUS_DIR / "ss-corpus-qxp.tsv")
    org = _org_split("spontaneous", "audio_file").select(
        "audio_file", "org_split", "remove_disfluency", "transcript_for_asr", "type"
    )
    has_tx = pl.col("transcription").fill_null("").str.len_chars() > 0
    validated = pl.col("split").is_not_null()
    df = corpus.join(org, on="audio_file", how="left").with_columns(
        clean=_null_if_empty(_norm("remove_disfluency").fill_null("")),
        raw=_null_if_empty(_norm("transcription").fill_null("")),
        silver_txt=pl.when(pl.col("type") == "silver").then(_norm("transcript_for_asr")),
    )
    return df.select(
        clip_id=pl.col("audio_file").str.replace(r"\.mp3$", ""),
        corpus=pl.lit("spontaneous"),
        speaker=pl.col("client_id"),
        duration_s=pl.col("duration_ms").cast(pl.Float64) / 1000,
        text=pl.when(has_tx).then(pl.coalesce("clean", "raw")),
        label=pl.when(has_tx & validated)
        .then(pl.lit("gold"))
        .when(has_tx)
        .then(pl.lit("pending"))
        .when(pl.col("silver_txt").is_not_null())
        .then(pl.lit("silver"))
        .otherwise(pl.lit("none")),
        silver=pl.col("silver_txt"),
        bucket=pl.when(has_tx & validated)
        .then(pl.lit("validated"))
        .when(has_tx)
        .then(pl.lit("pending"))
        .otherwise(pl.lit("untranscribed")),
        cv_split=pl.col("split"),
        org_split="org_split",
        sentence_id=pl.col("prompt_id"),
        prompt=_norm("prompt"),
        gender="gender",
        age="age",
        up_votes=pl.col("votes").cast(pl.Float64).cast(pl.Int32),
        down_votes=pl.lit(None, pl.Int32),
    )


def test_set() -> pl.DataFrame:
    meta = read_tsv(TEST_DIR / "qxp_test_dataset.tsv")
    return meta.select(
        clip_id=pl.col("file_name").str.replace(r"\.wav$", ""),
        corpus=pl.lit("test"),
        speaker=pl.lit(None, pl.String),
        duration_s=pl.lit(None, pl.Float64),
        text=pl.lit(None, pl.String),
        label=pl.lit("none"),
        silver=pl.lit(None, pl.String),
        bucket=pl.lit("test"),
        cv_split=pl.lit(None, pl.String),
        org_split=pl.lit(None, pl.String),
        sentence_id=pl.lit(None, pl.String),
        prompt=pl.lit(None, pl.String),
        gender="gender",
        age="age",
        up_votes=pl.lit(None, pl.Int32),
        down_votes=pl.lit(None, pl.Int32),
        test_type=pl.col("type"),
    )


def out_of_domain() -> pl.DataFrame:
    """Organizer radio-drama clips (Radio Onda Azul, Puno): broadcast speech, evaluation only."""
    meta = read_tsv(EXTERNAL / "ood_raw" / "additional_data_qxp.tsv")
    return meta.select(
        clip_id=pl.col("path").str.replace(r"\.wav$", ""),
        corpus=pl.lit("ood"),
        speaker=pl.lit(None, pl.String),
        duration_s=pl.lit(None, pl.Float64),
        text=_norm("sentence"),
        label=pl.lit("gold"),
        bucket=pl.lit("ood"),
    )


def segments() -> pl.DataFrame:
    """Pause-cut segments of long untranscribed spontaneous clips (``scripts/segment_long.py``)."""
    path = MANIFESTS / "segments.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path).select(
        "clip_id",
        corpus=pl.lit("spontaneous"),
        speaker="speaker",
        duration_s="duration_s",
        text=pl.lit(None, pl.String),
        label=pl.lit("none"),
        bucket=pl.lit("segment"),
        prompt="prompt",
        gender="gender",
        age="age",
        seg_wav="wav",
    )


def build() -> pl.DataFrame:
    """Concatenate all sources and attach WAV paths."""
    parts = [scripted(), spontaneous(), test_set(), out_of_domain(), segments()]
    df = pl.concat([p for p in parts if p.height], how="diagonal")
    wav = (
        pl.when(pl.col("corpus") == "test")
        .then(pl.lit(str(TEST_DIR / "audios") + "/") + pl.col("clip_id") + ".wav")
        .when(pl.col("corpus") == "ood")
        .then(pl.lit(str(AUDIO_OOD) + "/") + pl.col("clip_id") + ".wav")
        .when(pl.col("bucket") == "segment")
        .then(pl.col("seg_wav"))
        .otherwise(pl.lit(str(AUDIO) + "/") + pl.col("clip_id") + ".wav")
    )
    return df.with_columns(wav=wav).drop("seg_wav", strict=False)
