"""Cut long untranscribed spontaneous recordings into <= ~22 s segments at low-energy points.

432 untranscribed spontaneous clips (6 h, up to 1,716 s) exceed what the trainers (35 s) and the
omniASR LLM pipeline (40 s) accept. They are valuable for self-training: spontaneous speech from
speakers with many older voices (reports/eda/FINDINGS.md §1). We split each at the quietest 300 ms
window between ``min_s`` and ``max_s`` after the previous cut (energy-based segmentation, as
commonly used for long-form pseudo-labelling, e.g. Likhomanenko et al. 2021 slimIPL data prep),
so cuts fall in pauses rather than inside words.

Writes ``data/audio_seg/<clip>_sNN.wav`` and ``data/manifests/segments.parquet`` (merged into the
main manifest by ``scripts/build_manifest.py``; segments inherit their parent's speaker and role).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise

import polars as pl
import soundfile as sf
import typer

from qxp_asr.audio import load
from qxp_asr.paths import DATA, MANIFESTS, SAMPLE_RATE
from qxp_asr.segment import cut_points

SEG_DIR = DATA / "audio_seg"


def split(args: tuple[str, str, float, float]) -> list[dict]:
    clip_id, path, min_s, max_s = args
    wav = load(path)
    bounds = cut_points(wav, min_s, max_s)
    rows = []
    for k, (a, b) in enumerate(pairwise(bounds)):
        seg = wav[a:b]
        if len(seg) < SAMPLE_RATE:  # drop < 1 s tails
            continue
        out = SEG_DIR / f"{clip_id}_s{k:02d}.wav"
        sf.write(out, seg, SAMPLE_RATE, subtype="PCM_16")
        rows.append(
            {
                "clip_id": out.stem,
                "parent": clip_id,
                "wav": str(out),
                "duration_s": len(seg) / SAMPLE_RATE,
                "offset_s": a / SAMPLE_RATE,
            }
        )
    return rows


def main(
    min_dur: float = typer.Option(35.0, help="segment clips longer than this"),
    min_s: float = typer.Option(12.0),
    max_s: float = typer.Option(22.0),
    workers: int = typer.Option(4),
) -> None:
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    long = man.filter(
        (pl.col("corpus") == "spontaneous")
        & (pl.col("duration_s") > min_dur)
        & ~pl.col("label").is_in(["gold", "pending"])
    )
    jobs = [(c, w, min_s, max_s) for c, w in long.select("clip_id", "wav").iter_rows()]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for chunk in pool.map(split, jobs) for r in chunk]
    seg = pl.DataFrame(rows).join(
        long.select(
            pl.col("clip_id").alias("parent"), "speaker", "gender", "age", "prompt", "role"
        ),
        on="parent",
    )
    seg.write_parquet(MANIFESTS / "segments.parquet")
    print(
        f"{long.height} long clips ({long['duration_s'].sum() / 3600:.2f} h) -> {seg.height} "
        f"segments ({seg['duration_s'].sum() / 3600:.2f} h), max {seg['duration_s'].max():.1f} s"
    )


if __name__ == "__main__":
    typer.run(main)
