"""Benchmark every public Puno Quechua (qxp) ASR system on evaluation sets that are clean for all.

Sets (from the manifest, never the data-use-restricted ``test`` role):

``ood``
    The 109 organizer radio-drama clips (~16 min). Never used for training by us, by the
    organizers (they report it as their own out-of-domain set) or, as far as we know, by anyone.
``dev_spont39``
    ``dev_spont`` clips of at most 39 s (295 clips, 88 min). Secondary: the organizers' cv26
    checkpoints were trained on 307/310 of these clips, and community models trained on the
    Spontaneous Speech corpus may have seen them too.

Every system is scored identically: hypotheses go through ``qxp_asr.text.normalize`` (curated
orthography alphabet) and then ``qxp_asr.metrics.wer_cer`` (NFC, lowercase, punctuation stripped,
micro WER / CER with jiwer) against the manifest's curated reference.

Decoders (all CPU-only, fp32, ``torch.set_num_threads``; run them under ``nice``):

* ``omni``   omniASR CTC / LLM cards through ``omnilingual_asr``'s ``ASRInferencePipeline``
             (``lang=qxp_Latn``; the LLM pipeline default is beam size 1 = greedy). Run inside
             ``envs/omni``.
* ``hf``     Hugging Face checkpoints: ``Wav2Vec2ForCTC`` / ``Wav2Vec2BertForCTC`` (greedy CTC,
             one utterance at a time so padding never matters; MMS adapters via ``--target-lang``),
             Whisper (greedy, the checkpoint's own generation config, 30 s windows) and fairseq
             wav2vec2 CTC checkpoints (converted in memory with ``bench_organizer.fairseq_to_hf``).
* ``import`` existing per-clip prediction parquet files (our systems, organizer runs).

``score`` scores every run in ``--out-dir`` and ``table`` renders the markdown table.

Example (server B)::

    cd /workspace/simbig/envs/omni && CUDA_VISIBLE_DEVICES= nice -n 15 uv run python \
        bench_all_systems.py omni omniASR_CTC_1B_v2 --sets ood --out-dir runs
    cd /workspace/simbig/envs/bench_hf && export PYTHONPATH=/workspace/simbig/repo/src
    CUDA_VISIBLE_DEVICES= nice -n 15 .venv/bin/python bench_all_systems.py hf facebook/mms-1b-all \
        --name mms-1b-all-quz --target-lang quz --sets ood --out-dir runs
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from qxp_asr.audio import load
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import MANIFESTS, SAMPLE_RATE
from qxp_asr.text import normalize

app = typer.Typer(add_completion=False, no_args_is_help=True)

SETS = ("ood", "dev_spont39")
SPONT_MAX_S = 39.0
WHISPER_WINDOW_S = 30.0
OMNI_LANG = "qxp_Latn"
OutDir = Annotated[Path, typer.Option(help="Directory holding one parquet + json per run")]


# --------------------------------------------------------------------------- data


def eval_set(name: str, manifest: Path = MANIFESTS / "all.parquet") -> pl.DataFrame:
    """clip_id / wav / text / duration of one evaluation set, sorted by duration."""
    man = pl.read_parquet(manifest)
    if name == "ood":
        df = man.filter(pl.col("role") == "ood")
    elif name == "dev_spont39":
        df = man.filter((pl.col("role") == "dev_spont") & (pl.col("duration_s") <= SPONT_MAX_S))
    else:
        msg = f"unknown set {name!r}; choose from {SETS}"
        raise typer.BadParameter(msg)
    if (df["role"] == "test").any():  # data-use agreement: never touch the shared-task test
        msg = "test clips must never be evaluated here"
        raise RuntimeError(msg)
    return df.select("clip_id", "wav", "text", "duration_s").sort("duration_s")


def save_run(out_dir: Path, name: str, set_name: str, hyps: dict[str, str], **meta: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{name}__{set_name}"
    pl.DataFrame({"clip_id": list(hyps), "hyp": list(hyps.values())}).write_parquet(
        stem.with_suffix(".parquet")
    )
    stem.with_suffix(".json").write_text(json.dumps({"name": name, "set": set_name, **meta}))


def cpu_torch(threads: int) -> Any:
    """Import torch pinned to the CPU with ``threads`` intra-op threads."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch


def _timed_sets(
    sets: str, out_dir: Path, name: str, params: int, decode_fn: Any, **meta: Any
) -> None:
    for set_name in sets.split(","):
        df = eval_set(set_name)
        t0 = time.time()
        hyps = decode_fn(df)
        elapsed = time.time() - t0
        audio_s = float(df["duration_s"].sum())
        print(f"{name} {set_name}: {len(hyps)} clips, {elapsed:.0f} s, RTF {elapsed / audio_s:.3f}")
        save_run(
            out_dir, name, set_name, hyps, params=params, seconds=elapsed, audio_s=audio_s, **meta
        )
        quick = wer_cer(df["text"].to_list(), [normalize(hyps[c]) for c in df["clip_id"]])
        print(f"  WER {quick['wer']:.2f}  CER {quick['cer']:.2f}  (n={quick['n']})", flush=True)


# --------------------------------------------------------------------------- omniASR


@app.command()
def omni(
    card: str,
    name: str = typer.Option("", help="Run label (default: the card name)"),
    sets: str = typer.Option("ood"),
    batch_size: int = typer.Option(4),
    threads: int = typer.Option(4),
    out_dir: OutDir = Path("runs"),
) -> None:
    """Zero-shot omniASR (CTC or LLM card) with the official inference pipeline, on CPU."""
    torch = cpu_torch(threads)
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipe = ASRInferencePipeline(model_card=card, device="cpu", dtype=torch.float32)
    params = sum(p.numel() for p in pipe.model.parameters())

    def decode(df: pl.DataFrame) -> dict[str, str]:
        audio = [{"waveform": load(w), "sample_rate": SAMPLE_RATE} for w in df["wav"]]
        with torch.inference_mode():
            texts = pipe.transcribe(audio, lang=[OMNI_LANG] * len(audio), batch_size=batch_size)
        return dict(zip(df["clip_id"], texts, strict=True))

    _timed_sets(sets, out_dir, name or card, params, decode, model=card, decoder="pipeline")


# --------------------------------------------------------------------------- Hugging Face


def _snapshot(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    skip = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "runs/*", "*.msgpack", "*.h5"]
    if model == "facebook/mms-1b-all":  # 1,162 adapters: fetch only the base + Quechua ones
        allow = ["*.json", "model.safetensors", "adapter.qu*.safetensors"]
        return Path(snapshot_download(model, allow_patterns=allow))
    return Path(snapshot_download(model, ignore_patterns=skip))


def _hf_ctc(torch: Any, path: Path, target_lang: str | None) -> tuple[Any, int]:
    from transformers import AutoModelForCTC, AutoProcessor

    if target_lang:  # MMS: swap in the language adapter and its vocabulary
        processor = AutoProcessor.from_pretrained(path, target_lang=target_lang)
        model = AutoModelForCTC.from_pretrained(
            path, target_lang=target_lang, ignore_mismatched_sizes=True
        )
    else:
        processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForCTC.from_pretrained(path)
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    def decode(df: pl.DataFrame) -> dict[str, str]:
        hyps = {}
        with torch.inference_mode():
            for clip_id, wav in zip(df["clip_id"], df["wav"], strict=True):
                feats = processor(load(wav), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                logits = model(**feats).logits
                hyps[clip_id] = processor.decode(logits.argmax(-1)[0])
        return hyps

    return decode, params


def _hf_whisper(torch: Any, path: Path, language: str | None, batch: int) -> tuple[Any, int]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path).eval()
    params = sum(p.numel() for p in model.parameters())
    gen: dict[str, Any] = {"num_beams": 1, "do_sample": False}
    if language:
        gen |= {"language": language, "task": "transcribe"}
    step = int(WHISPER_WINDOW_S * SAMPLE_RATE)

    def decode(df: pl.DataFrame) -> dict[str, str]:
        segs = []
        for clip_id, wav_path in zip(df["clip_id"], df["wav"], strict=True):
            wav = load(wav_path)
            segs += [
                (clip_id, k, wav[i : i + step]) for k, i in enumerate(range(0, len(wav), step))
            ]
        parts: dict[str, dict[int, str]] = {}
        with torch.inference_mode():
            for b in range(0, len(segs), batch):
                chunk = segs[b : b + batch]
                feats = processor.feature_extractor(
                    [s[2] for s in chunk], sampling_rate=SAMPLE_RATE, return_tensors="pt"
                ).input_features
                texts = processor.batch_decode(
                    model.generate(feats, **gen), skip_special_tokens=True
                )
                for (clip_id, k, _), text in zip(chunk, texts, strict=True):
                    parts.setdefault(clip_id, {})[k] = text.strip()
        return {c: " ".join(p[k] for k in sorted(p)).strip() for c, p in parts.items()}

    return decode, params


def _fairseq_ctc(torch: Any, path: Path) -> tuple[Any, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_organizer import ctc_greedy, fairseq_to_hf

    model, vocab, norm = fairseq_to_hf(path / "checkpoint_best.pt")
    params = sum(p.numel() for p in model.parameters())
    blank = vocab.index("<s>")

    def decode(df: pl.DataFrame) -> dict[str, str]:
        hyps = {}
        with torch.inference_mode():
            for clip_id, wav in zip(df["clip_id"], df["wav"], strict=True):
                x = torch.from_numpy(load(wav)).float()
                if norm:  # fairseq task.normalize: per-utterance layer norm of the waveform
                    x = torch.nn.functional.layer_norm(x, x.shape)
                ids = model(x[None]).logits.argmax(-1)[0].tolist()
                hyps[clip_id] = ctc_greedy(ids, vocab, blank)
        return hyps

    return decode, params


@app.command()
def hf(
    model: str = typer.Argument(..., help="HF repo id or local snapshot directory"),
    name: str = typer.Option("", help="Run label (default: repo name)"),
    sets: str = typer.Option("ood"),
    target_lang: str = typer.Option("", help="MMS adapter language (e.g. quz, quy)"),
    language: str = typer.Option("", help="Whisper language override (default: gen. config)"),
    whisper_batch: int = typer.Option(8),
    threads: int = typer.Option(4),
    out_dir: OutDir = Path("runs"),
) -> None:
    """A Hugging Face checkpoint (CTC, Whisper or fairseq CTC), greedy decoding on CPU."""
    torch = cpu_torch(threads)
    path = _snapshot(model)
    if (path / "checkpoint_best.pt").exists():
        kind = "fairseq_ctc"
        decode, params = _fairseq_ctc(torch, path)
    elif json.loads((path / "config.json").read_text()).get("model_type") == "whisper":
        kind = "whisper"
        decode, params = _hf_whisper(torch, path, language or None, whisper_batch)
    else:
        kind = "ctc"
        decode, params = _hf_ctc(torch, path, target_lang or None)
    label = name or model.rstrip("/").split("/")[-1]
    print(f"{label}: {kind}, {params / 1e6:.0f}M params")
    _timed_sets(sets, out_dir, label, params, decode, model=model, decoder=f"{kind} greedy")


# --------------------------------------------------------------------------- existing predictions


@app.command(name="import")
def import_preds(
    paths: list[Path],
    prefix: str = typer.Option("", help="Prefix for run names (e.g. 'ours:')"),
    hyp_col: str = typer.Option("hyp"),
    out_dir: OutDir = Path("runs"),
) -> None:
    """Import per-clip prediction files (clip_id + hypothesis column) as runs on every set."""
    for path in paths:
        name = prefix + path.name.removesuffix(".parquet").removesuffix(".preds")
        pred = pl.read_parquet(path).select("clip_id", pl.col(hyp_col).alias("hyp"))
        for set_name in SETS:
            ids = eval_set(set_name).select("clip_id")
            sub = ids.join(pred, on="clip_id", how="inner")
            if sub.height == 0:
                continue
            hyps = dict(zip(sub["clip_id"], sub["hyp"].fill_null(""), strict=True))
            save_run(out_dir, name, set_name, hyps, source=str(path))
            print(f"{name} {set_name}: {sub.height}/{ids.height} clips")


# --------------------------------------------------------------------------- scoring


@app.command()
def score(out_dir: OutDir = Path("runs"), csv: Path = Path("scores.csv")) -> None:
    """Score every run in ``out_dir`` (normalized and raw hypotheses) into ``csv``."""
    refs = {s: eval_set(s).select("clip_id", "text") for s in SETS}
    rows = []
    for meta_path in sorted(out_dir.glob("*__*.json")):
        meta = json.loads(meta_path.read_text())
        pred = pl.read_parquet(meta_path.with_suffix(".parquet"))
        ref = refs[meta["set"]]
        df = ref.join(pred, on="clip_id", how="left").with_columns(pl.col("hyp").fill_null(""))
        hyps = df["hyp"].to_list()
        norm = wer_cer(df["text"].to_list(), [normalize(h) for h in hyps])
        raw = wer_cer(df["text"].to_list(), hyps)
        rows.append(
            {
                "name": meta["name"],
                "set": meta["set"],
                "n": norm["n"],
                "decoded": pred.join(ref, on="clip_id").height,
                "wer": round(norm["wer"], 2),
                "cer": round(norm["cer"], 2),
                "wer_raw": round(raw["wer"], 2),
                "cer_raw": round(raw["cer"], 2),
                "params_m": round(meta.get("params", 0) / 1e6),
                "rtf": round(meta["seconds"] / meta["audio_s"], 3) if "seconds" in meta else None,
            }
        )
    table = pl.DataFrame(rows).sort("set", "wer")
    table.write_csv(csv)
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(table)


@app.command()
def table(csv: Path = Path("scores.csv")) -> None:
    """Print a markdown pivot (one row per system, OOD and dev_spont39 columns)."""
    df = pl.read_csv(csv)
    wide = (
        df.with_columns(cell=pl.format("{} / {}", pl.col("wer"), pl.col("cer")))
        .pivot(on="set", index="name", values="cell")
        .join(df.filter(pl.col("set") == "ood").select("name", "wer"), on="name", how="left")
        .sort("wer", nulls_last=True)
    )
    print("| system | OOD WER / CER | dev_spont<=39 WER / CER |\n|---|---|---|")
    for row in wide.iter_rows(named=True):
        print(f"| {row['name']} | {row.get('ood') or '—'} | {row.get('dev_spont39') or '—'} |")


if __name__ == "__main__":
    app()
