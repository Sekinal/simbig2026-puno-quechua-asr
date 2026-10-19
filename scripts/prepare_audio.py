"""Convert every Common Voice qxp clip (scripted + spontaneous) to 16 kHz mono WAV.

Output: ``data/audio/<clip_id>.wav`` where clip_id is the mp3 stem. Parallel and resumable.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich.progress import track

from qxp_asr.audio import to_wav
from qxp_asr.paths import AUDIO, SCRIPTED_DIR, SPONTANEOUS_DIR


def main(workers: int = typer.Option(os.cpu_count() or 8)) -> None:
    sources = sorted((SCRIPTED_DIR / "clips").glob("*.mp3")) + sorted(
        (SPONTANEOUS_DIR / "audios").glob("*.mp3")
    )
    jobs: dict = {}
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for src in sources:
            jobs[pool.submit(to_wav, src, AUDIO / f"{src.stem}.wav")] = src
        done = track(as_completed(jobs), total=len(jobs), description="decoding")
        failures.extend((jobs[f], repr(f.exception())) for f in done if f.exception() is not None)
    print(f"converted {len(sources) - len(failures)}/{len(sources)} clips into {AUDIO}")
    for src, err in failures:
        print(f"FAILED {src}: {err}")


if __name__ == "__main__":
    typer.run(main)
