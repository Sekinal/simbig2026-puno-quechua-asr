"""omniASR (fairseq2) helpers: model loading, batched CTC posteriors, LLM transcription.

Only importable in the ``envs/omni`` environment (fairseq2 + omnilingual-asr). Numerically matches
``omnilingual_asr.models.inference.pipeline`` (Omnilingual ASR, arXiv:2511.09690): per-utterance
layer-norm of the waveform, zero padding, bf16 forward pass.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.batch_layout import BatchLayout
from torch.nn.functional import layer_norm

from qxp_asr.audio import load
from qxp_asr.segment import cut_points
from qxp_asr.vocab import SYMBOLS

LANG = "qxp_Latn"


@dataclass(slots=True)
class CtcOutput:
    clip_id: str
    log_probs: np.ndarray  # (frames, vocab) float16
    greedy_ids: np.ndarray  # collapsed, blank-free token ids


def length_batches(
    items: Sequence[tuple[str, str, float]], max_seconds: float
) -> Iterator[list[tuple[str, str, float]]]:
    """Group (clip_id, wav, duration) into batches with at most ``max_seconds`` of padded audio."""
    ordered = sorted(items, key=lambda x: x[2], reverse=True)
    batch: list[tuple[str, str, float]] = []
    for item in ordered:
        longest = batch[0][2] if batch else item[2]
        if batch and longest * (len(batch) + 1) > max_seconds:
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def load_ctc(card: str, device: str = "cuda") -> Wav2Vec2AsrModel:
    model = load_model(card, device=torch.device(device), dtype=torch.bfloat16)
    if not isinstance(model, Wav2Vec2AsrModel):
        msg = f"{card} is not a CTC model"
        raise TypeError(msg)
    return model.eval()


@torch.inference_mode()
def ctc_forward(
    model: Wav2Vec2AsrModel, wavs: list[np.ndarray], device: str = "cuda"
) -> list[np.ndarray]:
    """Return per-utterance CTC log-posteriors (float32 numpy, frames x vocab)."""
    tensors = [layer_norm(torch.from_numpy(w), w.shape) for w in wavs]
    lens = [t.shape[0] for t in tensors]
    seqs = torch.zeros(len(tensors), max(lens))
    for i, t in enumerate(tensors):
        seqs[i, : lens[i]] = t
    seqs = seqs.to(device, torch.bfloat16)
    layout = BatchLayout(seqs.shape, seq_lens=lens, device=seqs.device)
    logits, out_layout = model(seqs, layout)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return [logp[i, : out_layout.seq_lens[i]].cpu().numpy() for i in range(len(wavs))]


def greedy_collapse(log_probs: np.ndarray, blank: int = 0) -> np.ndarray:
    ids = log_probs.argmax(-1)
    keep = np.ones_like(ids, dtype=bool)
    keep[1:] = ids[1:] != ids[:-1]
    ids = ids[keep]
    return ids[ids != blank]


def run_ctc(
    model: Wav2Vec2AsrModel,
    items: Sequence[tuple[str, str, float]],
    max_seconds: float = 400.0,
    device: str = "cuda",
) -> Iterator[CtcOutput]:
    for batch in length_batches(items, max_seconds):
        wavs = [load(wav) for _, wav, _ in batch]
        for (clip_id, _, _), lp in zip(batch, ctc_forward(model, wavs, device), strict=True):
            yield CtcOutput(clip_id, lp.astype(np.float16), greedy_collapse(lp))


def omni_symbol_ids(card: str) -> list[int]:
    """omniASR token id for each symbol of ``qxp_asr.vocab.SYMBOLS`` (blank = 0)."""
    enc = load_tokenizer(card).create_encoder()
    ids = [0]
    for sym in SYMBOLS[1:]:
        # Encode between two letters: a bare space (or apostrophe) is stripped by the tokenizer.
        toks = enc(f"a{sym}a").tolist()
        if len(toks) != 3:
            msg = f"symbol {sym!r} is not a single omniASR token: {toks}"
            raise ValueError(msg)
        ids.append(toks[1])
    return ids


def prune_head(model: Wav2Vec2AsrModel, keep: list[int]) -> Wav2Vec2AsrModel:
    """Shrink the CTC output projection to ``keep`` rows (pretrained weights are copied)."""
    old = model.final_proj
    new = torch.nn.Linear(old.weight.shape[1], len(keep), bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight[keep])
        if old.bias is not None:
            new.bias.copy_(old.bias[keep])
    model.final_proj = new.to(old.weight.device, old.weight.dtype)
    return model


def load_qxp_ctc(
    card: str, checkpoint: str | None = None, device: str = "cuda", dtype=torch.bfloat16
) -> Wav2Vec2AsrModel:
    """omniASR CTC model with the head pruned to the qxp alphabet, optionally fine-tuned weights."""
    model = load_model(card, device=torch.device(device), dtype=torch.float32)
    prune_head(model, omni_symbol_ids(card))
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model.to(dtype).eval()


LLM_MAX_S = 39.0  # ASRInferencePipeline refuses audio longer than 40 s


def transcribe_llm(pipe, wavs: Sequence[np.ndarray], batch_size: int = 8) -> list[str]:
    """Transcribe with an omniASR LLM pipeline; clips over ``LLM_MAX_S`` are cut at pauses.

    Segment transcripts are concatenated (``qxp_asr.segment.cut_points``: 15-30 s pieces).
    """
    segs, owners = [], []
    for i, wav in enumerate(wavs):
        bounds = cut_points(wav, min_s=15.0, max_s=min(30.0, LLM_MAX_S))
        for a, b in pairwise(bounds):
            segs.append({"waveform": wav[a:b], "sample_rate": 16_000})
            owners.append(i)
    texts = pipe.transcribe(segs, lang=[LANG] * len(segs), batch_size=batch_size)
    out = [""] * len(wavs)
    for i, text in zip(owners, texts, strict=True):
        out[i] = f"{out[i]} {text}".strip()
    return out
