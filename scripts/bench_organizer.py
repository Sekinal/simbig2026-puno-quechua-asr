"""Benchmark the SIMBig 2026 organizers' released qxp ASR checkpoints on our evaluation roles.

Supports three checkpoint flavours (auto-detected from the downloaded HF snapshot directory):

* fairseq wav2vec2 CTC (``checkpoint_best.pt``, e.g. ``QuechuaBase/xls-r-cpt-qxp-cv26-validated``):
  converted in memory to ``transformers.Wav2Vec2ForCTC``. fairseq is *not* needed: the pickle is
  opened with stub classes and the target dictionary is read from the checkpoint's own
  ``task_state`` (the post-"Fix word-boundary token" cv26 dictionary, ``|`` = word boundary).
  Waveforms are layer-normalised per utterance (``task.normalize=True``) and decoded greedily
  (argmax, collapse repeats, drop the ``<s>`` blank) exactly like the organizers' viterbi eval.
* transformers ``Wav2Vec2ForCTC`` (``wav2vec2-base-qxp-finetuned``): processor + greedy decode.
* transformers Whisper (``whisper-base-qxp-finetuned``): greedy generation with the training
  prompt (language=spanish, task=transcribe); clips longer than 30 s are decoded in 30 s windows.

Scoring is the shared-task scorer (``qxp_asr.metrics.wer_cer``: NFC, lowercase, punctuation
stripped except the apostrophe, micro WER/CER with jiwer).

Run on the GPU box from an env with torch / transformers / omegaconf / polars / jiwer / typer,
with the repo sources importable (read-only), e.g.::

    PYTHONPATH=/workspace/simbig/repo/src uv run python bench_organizer.py run \
        checkpoints/xls-r-cpt-qxp-cv26-validated --name xls-r-cpt-qxp-cv26-validated \
        --out-dir out
    PYTHONPATH=/workspace/simbig/repo/src uv run python bench_organizer.py leakage
"""

from __future__ import annotations

import json
import pickle
import re
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import torch
import typer

from qxp_asr.audio import load
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import EXTERNAL, MANIFESTS, SAMPLE_RATE

app = typer.Typer(add_completion=False, no_args_is_help=True)

EVAL_ROLES = "dev_scripted,itest_scripted,dev_spont,itest_spont"
WHISPER_WINDOW_S = 30.0
_IGNORED_FAIRSEQ_KEYS = re.compile(r"(quantizer|project_q|final_proj|label_embs|target_glu)")
_CONV_TERM = re.compile(r"\[\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\](?:\s*\*\s*(\d+))?")


# --------------------------------------------------------------------------- fairseq loading


class _Stub:
    """Placeholder for any pickled ``fairseq.*`` object (e.g. ``fairseq.data.Dictionary``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__["init_args"] = (args, kwargs)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("fairseq"):
            return type(name, (_Stub,), {"__module__": module})
        return super().find_class(module, name)


_stub_pickle = types.ModuleType("stub_pickle")
_stub_pickle.Unpickler = _StubUnpickler  # type: ignore[attr-defined]
_stub_pickle.load = pickle.load  # type: ignore[attr-defined]


def load_fairseq_checkpoint(
    path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[str]]:
    """Return (encoder model cfg, state dict, target-dictionary symbols) without fairseq."""
    from omegaconf import OmegaConf

    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle)
    # cfg.model may be a plain dict holding DictConfig leaves: normalise to plain containers.
    cfg = OmegaConf.to_container(OmegaConf.create(ckpt["cfg"]["model"]), resolve=True)
    assert isinstance(cfg, dict)
    w2v_cfg = cfg["w2v_args"]["model"]
    w2v_cfg["normalize"] = cfg.get("normalize", w2v_cfg.get("normalize"))
    symbols = list(ckpt["task_state"]["target_dictionary"].symbols)
    return w2v_cfg, ckpt["model"], symbols


def parse_conv_layers(spec: str) -> list[tuple[int, int, int]]:
    """Parse fairseq's ``conv_feature_layers`` string (``"[(512, 10, 5)] + [(512, 3, 2)] * 4"``)."""
    layers: list[tuple[int, int, int]] = []
    for term in spec.split("+"):
        m = _CONV_TERM.fullmatch(term.strip())
        if m is None:
            msg = f"cannot parse conv_feature_layers term {term!r}"
            raise ValueError(msg)
        dim, kernel, stride, times = m.groups()
        layers += [(int(dim), int(kernel), int(stride))] * int(times or 1)
    return layers


_FE = "wav2vec2.feature_extractor.conv_layers."
_LAYER = "wav2vec2.encoder.layers."
# (fairseq Wav2VecCtc name without the ``w2v_encoder.w2v_model.`` prefix, Wav2Vec2ForCTC name)
_KEY_RULES = [
    (r"w2v_encoder\.proj\.(.+)", r"lm_head.\1"),
    (r"mask_emb", "wav2vec2.masked_spec_embed"),
    (r"feature_extractor\.conv_layers\.(\d+)\.0\.(.+)", rf"{_FE}\1.conv.\2"),
    # layer_norm mode: Sequential(TransposeLast, LayerNorm, ..) at .2.1; group mode: GroupNorm at .2
    (r"feature_extractor\.conv_layers\.(\d+)\.2\.(?:1\.)?(.+)", rf"{_FE}\1.layer_norm.\2"),
    (r"post_extract_proj\.(.+)", r"wav2vec2.feature_projection.projection.\1"),
    (r"layer_norm\.(.+)", r"wav2vec2.feature_projection.layer_norm.\1"),
    (r"encoder\.layer_norm\.(.+)", r"wav2vec2.encoder.layer_norm.\1"),
    (r"encoder\.pos_conv\.0\.(.+)", r"wav2vec2.encoder.pos_conv_embed.conv.\1"),
    (r"encoder\.layers\.(\d+)\.self_attn\.(.+)", rf"{_LAYER}\1.attention.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.(.+)", rf"{_LAYER}\1.layer_norm.\2"),
    (r"encoder\.layers\.(\d+)\.fc1\.(.+)", rf"{_LAYER}\1.feed_forward.intermediate_dense.\2"),
    (r"encoder\.layers\.(\d+)\.fc2\.(.+)", rf"{_LAYER}\1.feed_forward.output_dense.\2"),
    (r"encoder\.layers\.(\d+)\.final_layer_norm\.(.+)", rf"{_LAYER}\1.final_layer_norm.\2"),
]


def map_fairseq_key(key: str) -> str | None:
    """Map a fairseq ``Wav2VecCtc`` parameter name to its ``Wav2Vec2ForCTC`` name (None = drop)."""
    key = key.removeprefix("w2v_encoder.w2v_model.")
    if _IGNORED_FAIRSEQ_KEYS.search(key):
        return None
    for pattern, repl in _KEY_RULES:
        if m := re.fullmatch(pattern, key):
            return m.expand(repl)
    return None


def fairseq_to_hf(path: Path) -> tuple[Any, list[str], bool]:
    """Convert a fairseq wav2vec2 CTC checkpoint to ``Wav2Vec2ForCTC``.

    Returns (model, vocabulary, normalize_waveform).
    """
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    cfg, state, symbols = load_fairseq_checkpoint(path)
    layer_mode = cfg["extractor_mode"] == "layer_norm"
    conv = parse_conv_layers(cfg["conv_feature_layers"])
    config = Wav2Vec2Config(
        vocab_size=len(symbols),
        hidden_size=cfg["encoder_embed_dim"],
        num_hidden_layers=cfg["encoder_layers"],
        num_attention_heads=cfg["encoder_attention_heads"],
        intermediate_size=cfg["encoder_ffn_embed_dim"],
        hidden_act=cfg["activation_fn"],
        feat_extract_activation="gelu",
        feat_extract_norm="layer" if layer_mode else "group",
        do_stable_layer_norm=bool(cfg["layer_norm_first"]),
        conv_dim=[c[0] for c in conv],
        conv_kernel=[c[1] for c in conv],
        conv_stride=[c[2] for c in conv],
        conv_bias=bool(cfg["conv_bias"]),
        num_conv_pos_embeddings=cfg["conv_pos"],
        num_conv_pos_embedding_groups=cfg["conv_pos_groups"],
        layer_norm_eps=1e-5,
        apply_spec_augment=False,
        layerdrop=0.0,
        pad_token_id=symbols.index("<s>"),  # fairseq CTC blank = bos
        bos_token_id=symbols.index("<s>"),
        eos_token_id=symbols.index("</s>"),
    )
    model = Wav2Vec2ForCTC(config)
    target = model.state_dict()
    # Newer torch stores the weight-normed positional conv as parametrizations.
    weight_norm_alias = {
        "conv.weight_g": "conv.parametrizations.weight.original0",
        "conv.weight_v": "conv.parametrizations.weight.original1",
    }
    mapped: dict[str, torch.Tensor] = {}
    unused: list[str] = []
    for key, value in state.items():
        name = map_fairseq_key(key)
        if name is None:
            unused.append(key)
            continue
        if name not in target:
            for old, new in weight_norm_alias.items():
                name = name.replace(old, new)
        if name not in target or target[name].shape != value.shape:
            msg = f"{key} -> {name}: no matching target parameter / shape mismatch"
            raise KeyError(msg)
        mapped[name] = value
    missing = sorted(set(target) - set(mapped))
    if missing:
        msg = f"unconverted target parameters: {missing[:10]} ... ({len(missing)})"
        raise KeyError(msg)
    model.load_state_dict(mapped, strict=True)
    if unused:
        print(f"ignored {len(unused)} fairseq keys: {unused[:5]}")
    return model.eval(), symbols, bool(cfg["normalize"])


# --------------------------------------------------------------------------- inference


def length_batches(
    items: list[tuple[str, np.ndarray]], batch_seconds: float, max_items: int
) -> Iterator[list[tuple[str, np.ndarray]]]:
    """Group duration-sorted clips into batches capped by total padded audio and count."""
    batch: list[tuple[str, np.ndarray]] = []
    for item in sorted(items, key=lambda x: len(x[1])):
        padded = (len(batch) + 1) * len(item[1]) / SAMPLE_RATE
        if batch and (padded > batch_seconds or len(batch) >= max_items):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def ctc_greedy(ids: list[int], vocab: list[str], blank: int) -> str:
    """Collapse repeats, drop blanks; ``|`` and ``<unk>`` are word boundaries.

    ``<unk>`` doubles as the boundary for the pre-"Fix word-boundary token" checkpoints, whose
    dictionary had no ``|`` (as in the organizers' ``inference/transcribe.py``).
    """
    out, prev = [], None
    for i in ids:
        if i not in (prev, blank):
            out.append(vocab[i])
        prev = i
    boundary = {"|", "<unk>"}
    text = "".join(" " if t in boundary else ("" if t.startswith("<") else t) for t in out)
    return re.sub(r"\s+", " ", text).strip()


@torch.inference_mode()
def run_fairseq(
    model_dir: Path, items: list[tuple[str, np.ndarray]], device: str, batch_seconds: float
) -> dict[str, str]:
    model, vocab, normalize = fairseq_to_hf(model_dir / "checkpoint_best.pt")
    model.to(device)
    blank = vocab.index("<s>")
    hyps: dict[str, str] = {}
    for batch in length_batches(items, batch_seconds, max_items=64):
        lengths = [len(w) for _, w in batch]
        x = torch.zeros(len(batch), max(lengths))
        mask = torch.zeros(len(batch), max(lengths), dtype=torch.long)
        for j, (_, wav) in enumerate(batch):
            t = torch.from_numpy(wav).float()
            if normalize:  # fairseq: F.layer_norm(wav, wav.shape), eps=1e-5
                t = torch.nn.functional.layer_norm(t, t.shape)
            x[j, : len(t)] = t
            mask[j, : len(t)] = 1
        logits = model(x.to(device), attention_mask=mask.to(device)).logits
        frames = model._get_feat_extract_output_lengths(torch.tensor(lengths)).tolist()
        best = logits.argmax(-1).cpu()
        for j, (clip_id, _) in enumerate(batch):
            hyps[clip_id] = ctc_greedy(best[j, : frames[j]].tolist(), vocab, blank)
    return hyps


@torch.inference_mode()
def run_hf_ctc(
    model_dir: Path, items: list[tuple[str, np.ndarray]], device: str, batch_seconds: float
) -> dict[str, str]:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).eval().to(device)
    # Group-norm (base) models must not see padding: decode them one utterance at a time.
    group_norm = model.config.feat_extract_norm == "group"
    hyps: dict[str, str] = {}
    for batch in length_batches(items, batch_seconds, max_items=1 if group_norm else 64):
        feats = processor(
            [w for _, w in batch],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=not group_norm,
        )
        kwargs = {"attention_mask": feats.attention_mask.to(device)} if not group_norm else {}
        logits = model(feats.input_values.to(device), **kwargs).logits
        lengths = torch.tensor([len(w) for _, w in batch])
        frames = model._get_feat_extract_output_lengths(lengths).tolist()
        best = logits.argmax(-1).cpu()
        texts = processor.batch_decode([best[j, : frames[j]] for j in range(len(batch))])
        hyps.update({clip_id: t for (clip_id, _), t in zip(batch, texts, strict=True)})
    return hyps


@torch.inference_mode()
def run_whisper(
    model_dir: Path, items: list[tuple[str, np.ndarray]], device: str, batch_size: int
) -> dict[str, str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).eval().to(device)
    step = int(WHISPER_WINDOW_S * SAMPLE_RATE)
    segments = [
        (clip_id, k, wav[i : i + step])
        for clip_id, wav in items
        for k, i in enumerate(range(0, len(wav), step))
    ]
    segments.sort(key=lambda s: len(s[2]))
    parts: dict[str, dict[int, str]] = {}
    for b in range(0, len(segments), batch_size):
        chunk = segments[b : b + batch_size]
        feats = processor.feature_extractor(
            [s[2] for s in chunk], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device)
        out = model.generate(
            feats, language="spanish", task="transcribe", num_beams=1, do_sample=False
        )
        texts = processor.batch_decode(out, skip_special_tokens=True)
        for (clip_id, k, _), text in zip(chunk, texts, strict=True):
            parts.setdefault(clip_id, {})[k] = text.strip()
    return {c: " ".join(p[k] for k in sorted(p)).strip() for c, p in parts.items()}


def detect_kind(model_dir: Path) -> str:
    if (model_dir / "checkpoint_best.pt").exists():
        return "fairseq"
    config = json.loads((model_dir / "config.json").read_text())
    return "whisper" if config.get("model_type") == "whisper" else "hf_ctc"


# --------------------------------------------------------------------------- scoring


def score(pred: pl.DataFrame) -> pl.DataFrame:
    """Micro WER/CER per role, plus the unseen-sentence subset of scripted roles."""
    rows = []
    for role, g in pred.group_by("role", maintain_order=True):
        subsets = [("all", g)]
        if g["unseen_sentence"].any():
            subsets.append(("unseen", g.filter(pl.col("unseen_sentence"))))
        for subset, s in subsets:
            m = wer_cer(s["text"].to_list(), s["hyp"].to_list())
            rows.append({"role": role[0], "subset": subset, **m})
    return pl.DataFrame(rows).sort("role", "subset")


@app.command()
def run(
    model_dir: Path,
    name: str = typer.Option(..., help="Label used for output files"),
    roles: str = typer.Option(EVAL_ROLES),
    out_dir: Annotated[Path, typer.Option()] = Path("out"),
    device: str = typer.Option("cuda"),
    batch_seconds: float = typer.Option(120.0, help="Max padded audio seconds per CTC batch"),
    whisper_batch: int = typer.Option(16),
    max_gpu_gb: float = typer.Option(8.0, help="Hard cap on this process's CUDA memory"),
    threads: int = typer.Option(4, help="CPU threads (when --device cpu)"),
) -> None:
    """Transcribe the manifest roles with one organizer checkpoint and score them."""
    role_list = roles.split(",")
    if "test" in role_list:
        msg = "role 'test' is under a data-use agreement; benchmark it separately"
        raise typer.BadParameter(msg)
    torch.set_num_threads(threads)
    if device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_gb * 2**30 / total))
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("role").is_in(role_list))
    items = [(c, load(w)) for c, w in man.select("clip_id", "wav").iter_rows()]
    kind = detect_kind(model_dir)
    print(f"{name}: {kind}, {len(items)} clips, {man['duration_s'].sum() / 3600:.2f} h")
    t0 = time.time()
    if kind == "fairseq":
        hyps = run_fairseq(model_dir, items, device, batch_seconds)
    elif kind == "hf_ctc":
        hyps = run_hf_ctc(model_dir, items, device, batch_seconds)
    else:
        hyps = run_whisper(model_dir, items, device, whisper_batch)
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    print(f"decoded in {elapsed:.0f}s, peak CUDA memory {peak:.2f} GiB")

    pred = man.select("clip_id", "role", "speaker", "unseen_sentence", "text").join(
        pl.DataFrame({"clip_id": list(hyps), "hyp": list(hyps.values())}), on="clip_id"
    )
    if pred.height != man.height:
        msg = f"only {pred.height}/{man.height} clips decoded"
        raise RuntimeError(msg)
    scores = score(pred).with_columns(model=pl.lit(name))
    out_dir.mkdir(parents=True, exist_ok=True)
    # published with the code: no speaker ids (Common Voice terms: no re-identification)
    pred.drop("speaker").write_parquet(out_dir / f"{name}.preds.parquet")
    scores.write_csv(out_dir / f"{name}.scores.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=160):
        print(scores)


# --------------------------------------------------------------------------- leakage


def _clip_stem(col: pl.Expr) -> pl.Expr:
    return col.str.replace(r"\.(mp3|wav)$", "")


@app.command()
def leakage(
    splits_dir: Annotated[Path, typer.Option()] = EXTERNAL / "splits_cv26",
    roles: str = typer.Option(EVAL_ROLES),
) -> None:
    """Speaker- and clip-level overlap of our eval roles with the organizers' train splits."""
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("role").is_in(roles.split(",")))
    train = {}
    for corpus, clip_col in (("scripted", "path"), ("spontaneous", "audio_file")):
        df = pl.read_csv(
            splits_dir / corpus / "train.tsv", separator="\t", quote_char=None, infer_schema=False
        )
        train[corpus] = df.select(
            "client_id", _clip_stem(pl.col(clip_col)).alias("clip_id")
        ).unique()
    rows = []
    for role, g in man.group_by("role", maintain_order=True):
        speakers = set(g["speaker"].drop_nulls())
        clips = set(g["clip_id"])
        row: dict[str, Any] = {"role": role[0], "clips": g.height, "speakers": len(speakers)}
        for corpus, t in train.items():
            leaked = speakers & set(t["client_id"])
            clips_by_leaked = g.filter(pl.col("speaker").is_in(list(leaked))).height
            row[f"{corpus}_train_speakers"] = len(leaked)
            row[f"{corpus}_train_speaker_clips"] = clips_by_leaked
            row[f"{corpus}_train_same_clip"] = len(clips & set(t["clip_id"]))
        rows.append(row)
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250):
        print(pl.DataFrame(rows).sort("role"))


# --------------------------------------------------------------------------- summary


def organizer_refs(splits_dir: Path) -> pl.DataFrame:
    """clip_id -> the organizers' own training/eval transcript (``transcript_for_asr``)."""
    frames = []
    for corpus, clip_col in (("scripted", "path"), ("spontaneous", "audio_file")):
        for split in ("train", "dev", "test"):
            df = pl.read_csv(
                splits_dir / corpus / f"{split}.tsv",
                separator="\t",
                quote_char=None,
                infer_schema=False,
            )
            frames.append(
                df.select(
                    _clip_stem(pl.col(clip_col)).alias("clip_id"),
                    pl.col("transcript_for_asr").alias("org_ref"),
                )
            )
    return pl.concat(frames).unique("clip_id")


@app.command()
def summarize(
    out_dir: Annotated[Path, typer.Option()] = Path("out"),
    splits_dir: Annotated[Path, typer.Option()] = EXTERNAL / "splits_cv26",
) -> None:
    """Score every ``*.preds.parquet`` against our references and the organizers' own ones."""
    refs = organizer_refs(splits_dir)
    # Re-read role / reference / unseen flag from the current manifest so every model is scored
    # against the same labels even if the manifest was rebuilt between decoding runs.
    labels = pl.read_parquet(MANIFESTS / "all.parquet").select(
        "clip_id", "role", "speaker", "unseen_sentence", "text"
    )
    rows = []
    for path in sorted(out_dir.glob("*.preds.parquet")):
        model = path.name.removesuffix(".preds.parquet")
        pred = (
            pl.read_parquet(path)
            .select("clip_id", "hyp")
            .join(labels, on="clip_id", how="inner")
            .join(refs, on="clip_id", how="left")
        )
        org = pred.filter(pl.col("org_ref").is_not_null()).with_columns(text=pl.col("org_ref"))
        for ref_name, df in (("ours", pred), ("organizers", org)):
            rows += score(df).with_columns(model=pl.lit(model), ref=pl.lit(ref_name)).to_dicts()
    table = pl.DataFrame(rows).select("model", "ref", "role", "subset", "n", "wer", "cer")
    table.write_csv(out_dir / "summary.csv")
    with pl.Config(tbl_rows=200, tbl_width_chars=200, float_precision=2):
        print(table)


if __name__ == "__main__":
    app()
