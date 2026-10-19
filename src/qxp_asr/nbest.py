"""N-best generation and sequence log-likelihoods for omniASR LLM-ASR models.

``NBestBeamSearch.generate_nbest`` is adapted from
``Wav2Vec2LlamaBeamSearchSeq2SeqGenerator.generate_hypotheses_one_segment`` in
facebookresearch/omnilingual-asr (Copyright 2025 Meta Platforms, Inc. and affiliates; Apache License
2.0). Change: return *all* beams and their scores instead of only the top hypothesis, so they can
serve as the candidate group for sequence-level training (MWER / GSPO) and rescoring. Only
non-streaming models are supported (omniASR_LLM_*_v2).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.nn import BatchLayout, IncrementalStateBag
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
from torch import Tensor

from qxp_asr.omni import LANG


class NBestBeamSearch(Wav2Vec2LlamaBeamSearchSeq2SeqGenerator):
    def __init__(
        self, model: Wav2Vec2LlamaModel, nbest: int, temperature: float | None = None
    ) -> None:
        """``temperature`` set: ``nbest`` independent rows per example instead of beam search
        (whose hypotheses often differ only in tokenisation): ancestral samples at that temperature,
        or greedy rows for ``temperature=0``. Independent rows also accept ``forced`` token prefixes
        (one per row) and can ``record`` per-step entropy and runner-up tokens (fork exploration).
        """
        super().__init__(
            model=model,
            config=Wav2Vec2LlamaBeamSearchConfig(nbest=nbest, length_norm=False),
            streaming_config=model.streaming_config,
        )
        self.temperature = temperature
        self.forced: list[list[int]] | None = None
        self.record = False
        self.trace: list[list[tuple[float, int, int]]] = []

    @torch.inference_mode()
    def generate_nbest(self, batch: Seq2SeqBatch) -> tuple[list[list[list[int]]], Tensor]:
        """Return per example the ``nbest`` token sequences (no BOS/EOS/pad) and their scores."""
        model = self.model
        ctx, ctx_lens, _ = model(batch, return_decoder_inputs=True)
        ctx, ctx_lens = ctx[0], ctx_lens[0]
        b = ctx.size(0)
        device, dtype, n = ctx.device, ctx.dtype, self.config.nbest
        sep = torch.arange(b, device=device).unsqueeze(1) * n
        max_len = model.max_generation_length
        inputs = torch.zeros([b * n, max_len, model.model_dim], device=device, dtype=dtype)
        inputs[:, : ctx.size(1)] = ctx.repeat_interleave(n, dim=0)
        ctx_len = torch.tensor(ctx_lens, device=device).repeat_interleave(n)
        out = torch.full_like(inputs[:, :, 0], fill_value=self.pad_idx, dtype=torch.int)
        scores = torch.zeros_like(inputs[:, 0, 0], dtype=torch.float)
        if self.temperature is None:
            scores -= 1e6
            scores[::n] = 0.0
        state = IncrementalStateBag(max_num_steps=max_len)
        start = int(ctx_len.min()) - 1
        model.llama_decoder(
            seqs=inputs[:, :start],
            seqs_layout=BatchLayout.of(inputs[:, :start], [start] * b * n),
            state_bag=state,
        )
        state.increment_step_nr(start)
        eos = torch.zeros_like(ctx_len, dtype=torch.bool)
        t = start
        rows = torch.arange(b * n, device=device)
        if self.forced is not None:
            flen = torch.tensor([len(f) for f in self.forced], device=device)
            fmat = torch.full((b * n, max(1, int(flen.max()))), self.pad_idx, device=device)
            for r, f in enumerate(self.forced):
                if f:
                    fmat[r, : len(f)] = torch.tensor(f, device=device)
        self.trace = [[] for _ in range(b * n)]
        while True:
            step = inputs[:, t : t + 1]
            dec = model.llama_decoder(
                seqs=step, seqs_layout=BatchLayout.of(step, [1] * b * n), state_bag=state
            )
            state.increment_step_nr(1)
            logp = torch.log_softmax(model.final_proj(dec).squeeze(1).float(), dim=-1)
            if self.temperature is None:
                cand = scores.unsqueeze(1) + logp
                cand[eos] = -torch.inf
                cand[eos, self.eos_idx] = scores[eos]
                top_scores, top_idx = cand.view(b, -1).topk(k=n, dim=-1, sorted=True)
                parent, token = self.idx_1d_to_2d(top_idx, cand.size(-1))
                parent = (parent + sep).view(-1)
                out, eos, scores = out[parent], eos[parent], scores[parent]
                state.reorder(parent)
                scores = torch.where(eos, scores, top_scores.view(-1))
                out[:, t] = token.view(-1)
            else:  # every row is its own sample: no reordering
                if self.temperature > 0:
                    token = torch.multinomial(torch.softmax(logp / self.temperature, -1), 1)
                    token = token.squeeze(1)
                else:
                    token = logp.argmax(-1)
                rel = t - (ctx_len - 1)  # position within the generated transcript
                if self.forced is not None:
                    use = (rel >= 0) & (rel < flen)
                    token = torch.where(use, fmat[rows, rel.clamp(0, fmat.size(1) - 1)], token)
                if self.record:
                    ent = -(logp.exp() * logp).sum(-1)
                    second = logp.topk(2, dim=-1).indices[:, 1]
                    live = ((rel >= 0) & ~eos).tolist()
                    for r in [i for i, v in enumerate(live) if v]:
                        self.trace[r].append((float(ent[r]), int(second[r]), int(token[r])))
                token = token.masked_fill(eos, self.eos_idx)
                gained = logp.gather(1, token.unsqueeze(1)).squeeze(1)
                scores = torch.where(eos, scores, scores + gained)
                out[:, t] = token
            waiting = t < ctx_len - 1  # longer contexts are still being consumed
            out[waiting, t] = self.pad_idx
            scores[waiting] = 0.0
            out[eos, t] = self.eos_idx
            new = out[:, t : t + 1]
            eos = (new == self.eos_idx).squeeze(1)
            emb = model.embed_text(new, dtype=dtype)
            inputs[~waiting, t + 1] = emb[~waiting].to(inputs.dtype).squeeze(1)
            t += 1
            if bool(eos.all()) or t >= max_len - 4:
                break
        special = {self.pad_idx, self.bos_idx, self.eos_idx}
        seqs = [[int(x) for x in row.tolist() if int(x) not in special] for row in out]
        grouped = [seqs[i * n : (i + 1) * n] for i in range(b)]
        return grouped, scores.view(b, n)


def make_batch(
    wavs: list[Tensor], targets: list[list[int]], pad_idx: int, device: str
) -> Seq2SeqBatch:
    """Seq2SeqBatch of layer-normalized waveforms and token targets (lang = qxp_Latn)."""
    lens = [len(w) for w in wavs]
    src = torch.zeros(len(wavs), max(lens))
    for i, w in enumerate(wavs):
        src[i, : len(w)] = torch.nn.functional.layer_norm(w, w.shape)
    tlens = [max(1, len(t)) for t in targets]
    tgt = torch.full((len(targets), max(tlens)), pad_idx, dtype=torch.long)
    for i, t in enumerate(targets):
        tgt[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return Seq2SeqBatch(
        source_seqs=src.to(device, torch.bfloat16),
        source_seq_lens=lens,
        target_seqs=tgt.to(device),
        target_seq_lens=tlens,
        example={"lang": [LANG] * len(wavs)},
    )


def sequence_logprobs(model: Wav2Vec2LlamaModel, batch: Seq2SeqBatch) -> Tensor:
    """Sum of token log-probabilities (incl. EOS) of each target sequence: shape (B,)."""
    _, logits, layout, _, ctx_lens, _ = model(batch, return_logits=True)
    targets, target_layout = batch.as_target_input()
    vi = model.target_vocab_info
    targets, target_layout = Wav2Vec2LlamaModel.add_eos(
        targets, target_layout, vi.pad_idx, vi.eos_idx
    )
    logits = Wav2Vec2LlamaModel.remove_context_logits(
        logits=logits,
        logit_layout=layout,
        targets=targets,
        target_layout=target_layout,
        decoder_context_seq_lens=ctx_lens,
    )
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = targets != vi.pad_idx
    return (tok * mask).sum(dim=1)


@contextlib.contextmanager
def cached_audio(
    model: Wav2Vec2LlamaModel, seqs: Tensor, seq_lens: list[int], repeat: int
) -> Iterator[None]:
    """Encode audio once; serve ``repeat`` copies per clip to the next forward pass.

    Lets a group of hypotheses (or candidates) of the same clip share one pass of the 7B encoder.
    """
    with torch.no_grad():
        emb, lens = model.embed_audio(seqs, seq_lens)
    emb = emb.repeat_interleave(repeat, dim=0)
    lens = [x for x in lens for _ in range(repeat)]
    model.embed_audio = lambda *_a, **_k: (emb, lens)  # instance attribute shadows the method
    try:
        yield
    finally:
        del model.embed_audio


def generate_forks(model: Wav2Vec2LlamaModel, batch: Seq2SeqBatch, m: int) -> list[list[list[int]]]:
    """Greedy transcript + ``m`` forks per example (exploration at uncertain tokens).

    Branching at the few high-entropy "forking" tokens drives most of the learning in LLM RL
    (Wang et al., 2025, "Beyond the 80/20 Rule", arXiv:2506.01939). Decode greedily while recording
    the per-step entropy, then for each of the ``m`` most uncertain positions force the runner-up
    token there and continue greedily. Returns per example [greedy, fork_1, ..., fork_m].
    """
    greedy = NBestBeamSearch(model, 1, temperature=0.0)
    greedy.record = True
    first, _ = greedy.generate_nbest(batch)
    prefixes = []
    for trace in greedy.trace:
        toks = [tok for _, _, tok in trace]
        by_entropy = sorted(range(len(trace)), key=lambda i: -trace[i][0])[:m]
        forks = [[*toks[:i], trace[i][1]] for i in by_entropy]
        prefixes += forks + [toks[:-1]] * (m - len(forks))  # short transcripts: repeat greedy
    fork = NBestBeamSearch(model, m, temperature=0.0)
    fork.forced = prefixes
    rest, _ = fork.generate_nbest(batch)
    return [f + r for f, r in zip(first, rest, strict=True)]
