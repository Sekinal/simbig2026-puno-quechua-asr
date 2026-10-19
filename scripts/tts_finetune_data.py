"""Write OmniVoice fine-tuning manifests (JSONL) from the provided labelled training clips.

OmniVoice zero-shot cannot tell near-miss Quechua transcripts apart (``tts_score.py`` pairwise
accuracy ~50% on OOD), so it is LoRA fine-tuned on our own data (provided data only): role
``train`` clips with a transcript, text normalised exactly as the ASR candidates it will score.
Dev / held-out / OOD / test clips are never included. A small slice of the training clips is the
recipe's dev set (loss monitoring only).

    uv run python scripts/tts_finetune_data.py --out /workspace/simbig/ext/OmniVoice/data/qxp
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

from qxp_asr.metrics import official_normalize
from qxp_asr.paths import MANIFESTS


def main(out: Path = typer.Option(...), n_dev: int = 300, seed: int = 0) -> None:
    man = (
        pl.read_parquet(MANIFESTS / "all.parquet")
        .filter((pl.col("role") == "train") & pl.col("text").is_not_null() & pl.col("wav_exists"))
        .with_columns(text=pl.col("text").map_elements(official_normalize, return_dtype=pl.Utf8))
        .filter(pl.col("text").str.len_chars() > 0)
        .sample(fraction=1.0, shuffle=True, seed=seed)
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("dev", man.head(n_dev)), ("train", man.slice(n_dev))]:
        with (out / f"{name}.jsonl").open("w") as f:
            for cid, wav, text in part.select("clip_id", "wav", "text").iter_rows():
                rec = {"id": cid, "audio_path": wav, "text": text, "language_id": "qxp"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        hours = part["duration_s"].sum() / 3600
        print(f"{name}: {part.height} clips, {hours:.1f} h -> {out / name}.jsonl")


if __name__ == "__main__":
    typer.run(main)
