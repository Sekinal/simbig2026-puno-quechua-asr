"""Text-only language-model loss for the LLaMA decoder of omniASR LLM-ASR (decoder text CPT).

The decoder writes the words, so in-language text adapts exactly the component that makes
orthographic and morphological errors on unseen sentences. Text-only adaptation of the LM part of
speech / LLM-ASR models is a known domain-adaptation route (internal-LM / text-only adaptation of
end-to-end ASR); we apply it through the same decoder LoRA, interleaved with ASR batches so the
decoder keeps its audio conditioning. Sequences are ``<bos> text <eos>`` with no audio prefix.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from fairseq2.nn.batch_layout import BatchLayout


class TextSampler:
    """Uniformly samples normalized lines from a text corpus and tokenizes them."""

    def __init__(self, path: Path, encoder, max_tokens: int = 200, seed: int = 0) -> None:
        lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
        self.lines = [ln for ln in lines if 2 <= len(ln.split()) <= 60]
        self.encode, self.max_tokens = encoder, max_tokens
        self.rng = random.Random(seed)

    def __call__(self, n: int) -> list[list[int]]:
        picks = self.rng.sample(self.lines, n)
        return [self.encode(t).tolist()[: self.max_tokens] for t in picks]


def text_lm_loss(model, token_lists: list[list[int]], device: str = "cuda") -> torch.Tensor:
    """Mean next-token cross-entropy of ``<bos> tokens <eos>`` under the decoder (no audio)."""
    vi = model.target_vocab_info
    seqs = [[vi.bos_idx, *t, vi.eos_idx] for t in token_lists]
    lens = [len(s) - 1 for s in seqs]
    inp = torch.full((len(seqs), max(lens)), vi.pad_idx, dtype=torch.long)
    tgt = torch.full_like(inp, vi.pad_idx)
    for i, s in enumerate(seqs):
        inp[i, : lens[i]] = torch.tensor(s[:-1])
        tgt[i, : lens[i]] = torch.tensor(s[1:])
    inp, tgt = inp.to(device), tgt.to(device)
    emb = model.embed_text(inp, dtype=torch.bfloat16)
    dec = model.llama_decoder(emb, BatchLayout.of(emb, lens))
    logits = model.final_proj(dec).float()
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=vi.pad_idx
    )
