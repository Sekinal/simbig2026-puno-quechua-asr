"""Synthesise Quechua speech from text with MMS-TTS (back-translation for ASR, step 1).

ASR "back-translation" (Hayashi et al., 2018; Rosenberg et al., 2019; Rossenbach et al., 2020)
turns text-only data into paired data by synthesising speech for it. There is no Puno (qxp) voice,
but MMS-TTS has Cusco Quechua (``facebook/mms-tts-quz``; Pratap et al., 2023; VITS, CC-BY-NC 4.0),
the closest variety (Southern Quechua, Cusco-Collao).

Writes 16 kHz WAVs and ``manifest.parquet`` (clip_id, wav, duration_s, text, target) to
``artifacts/tts/<name>/``; ``scripts/tts_roundtrip.py`` transcribes them with our ASR, and the
manifest can be passed to ``train_llm.py --pseudo name=.../manifest.parquet`` as training data.
Each utterance gets a random speaking rate (``--rate-min``..``--rate-max``): the single MMS voice
is otherwise monotonous, and slower speech is closer to the elderly test speakers.

Runs in an environment with transformers (``envs/orgbench``):
    uv run python scripts/tts_synth.py --name dev_unseen --roles dev_scripted --unseen-only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
import torch
import typer
from transformers import AutoTokenizer, VitsModel

ROOT = Path("/workspace/simbig")


def main(
    name: str = typer.Option(...),
    roles: str = typer.Option("dev_scripted"),
    unseen_only: bool = typer.Option(False),
    text_file: Path | None = typer.Option(None, help="one sentence per line instead of --roles"),
    limit: int = typer.Option(0),
    sample: int = typer.Option(0, help="random sample of this many sentences from --text-file"),
    min_words: int = typer.Option(3),
    max_words: int = typer.Option(25),
    rate_min: float = typer.Option(1.0),
    rate_max: float = typer.Option(1.0),
    gpu_mem_frac: float = typer.Option(0.05, help="cap: share the GPU with a running job"),
    card: str = typer.Option("facebook/mms-tts-quz"),
    seed: int = typer.Option(0),
) -> None:
    if text_file:
        texts = [t.strip() for t in text_file.read_text().splitlines() if t.strip()]
        texts = [t for t in texts if min_words <= len(t.split()) <= max_words]
        if sample:
            texts = pl.Series(texts).unique().sample(sample, seed=seed).to_list()
    else:
        man = pl.read_parquet(ROOT / "data" / "manifests" / "all.parquet").filter(
            pl.col("role").is_in(roles.split(",")) & pl.col("text").is_not_null()
        )
        if unseen_only:
            man = man.filter(pl.col("unseen_sentence"))
        texts = man["text"].unique(maintain_order=True).to_list()
    texts = texts[:limit] if limit else texts
    out = ROOT / "artifacts" / "tts" / name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_mem_frac)
    tok, model = AutoTokenizer.from_pretrained(card), VitsModel.from_pretrained(card).to(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for i, text in enumerate(texts):
        inputs = tok(text.lower(), return_tensors="pt").to(device)
        model.speaking_rate = float(rng.uniform(rate_min, rate_max))
        with torch.no_grad():
            wav = model(**inputs).waveform[0]
        if model.config.sampling_rate != 16_000:  # MMS-TTS voices are 16 kHz
            raise ValueError(f"unexpected sampling rate {model.config.sampling_rate}")
        wav = wav.cpu().numpy().astype(np.float32)
        path = out / f"tts_{name}_{i:06d}.wav"
        sf.write(path, wav, 16_000)
        rows.append(
            {
                "clip_id": path.stem,
                "wav": str(path),
                "duration_s": len(wav) / 16_000,
                "text": text,
                "target": text,
            }
        )
    pl.DataFrame(rows).write_parquet(out / "manifest.parquet")
    print(f"{len(rows)} utterances, {sum(r['duration_s'] for r in rows) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    typer.run(main)
