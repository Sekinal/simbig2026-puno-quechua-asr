"""TTS-likelihood verifier: how well does each candidate transcript explain the REAL audio?

Analysis-by-synthesis without synthesis (in the spirit of READ, arXiv:2606.04680): OmniVoice
(k2-fsa/OmniVoice, arXiv:2604.00688; a masked "diffusion LM" TTS over 8-codebook audio tokens that
lists Puno Quechua, qxp) is conditioned on a candidate text and asked to reconstruct masked tokens
of the clip's own audio. The unmasked half carries the speaker's voice; the masked half must be
predicted from the text. The same random masks are used for every candidate of a clip, so score
differences come from the text only:

    lp_tts(candidate) = - mean over K masks of the codebook-weighted cross-entropy of the masked
                          real audio tokens given (language qxp, candidate text, visible tokens)
    lp_tts0(candidate) = the same for the first (coarsest, most phonetic) codebook only

This is an acoustic-linguistic judge independent of our omniASR models. Caveat: OmniVoice's 9.8 h of
qxp training data probably come from Common Voice Puno (our dev/itest clips); OOD radio drama is the
clean check. Runs in OmniVoice's own environment (/workspace/simbig/ext/OmniVoice/.venv):

    python scripts/tts_score.py --cands /workspace/simbig/artifacts/cands/pool_dev_aug.parquet \
        --manifest /workspace/simbig/data/manifests/all.parquet --out .../tts_pool_dev_aug.parquet
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import soundfile as sf
import torch
import torchaudio.functional as taf
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.lora import load_lora_adapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--span", type=int, default=0, help="contiguous span masks of this many tokens")
    ap.add_argument("--roles", default="")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default="", help="OmniVoice LoRA checkpoint dir (exp29)")
    args = ap.parse_args()

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
    if args.lora:  # exp29 fine-tuned adapter (checkpoint dir), merged in memory
        model = load_lora_adapter(model, args.lora)
    model.eval()
    cfg = model.config
    cands = pl.read_parquet(args.cands).select("clip_id", "text")
    man = pl.read_parquet(args.manifest).select("clip_id", "wav", "role")
    man = man.filter(pl.col("clip_id").is_in(cands["clip_id"].unique()))
    if args.roles:
        man = man.filter(pl.col("role").is_in(args.roles.split(",")))
    if args.max_clips:
        man = man.head(args.max_clips)
    by_clip = cands.filter(pl.col("clip_id").is_in(man["clip_id"])).partition_by(
        "clip_id", as_dict=True
    )
    hop = model.audio_tokenizer.config.hop_length
    rows, t0 = [], time.time()
    for n, (cid, wav_path) in enumerate(man.select("clip_id", "wav").iter_rows()):
        wav, sr = sf.read(wav_path, dtype="float32")
        wav = taf.resample(torch.from_numpy(wav), sr, model.sampling_rate)
        wav = wav[: len(wav) - len(wav) % hop]
        rms = float(wav.pow(2).mean().sqrt())
        if 0 < rms < 0.1:
            wav = wav * 0.1 / rms
        with torch.inference_mode():
            codes = model.audio_tokenizer.encode(
                wav.view(1, 1, -1).to(model.audio_tokenizer.device)
            ).audio_codes.squeeze(0)  # (C, T)
        n_tok = codes.shape[-1]
        g = torch.Generator().manual_seed(args.seed + n)
        if args.span:  # K interleaved partitions of contiguous spans: every token masked once
            start = int(torch.randint(args.span * args.masks, (1,), generator=g))
            block = (torch.arange(n_tok) + start) // args.span % args.masks
            masks = [block == k for k in range(args.masks)]
        else:
            masks = [torch.rand(n_tok, generator=g) < args.ratio for _ in range(args.masks)]
        texts = [t for t in by_clip[(cid,)]["text"].to_list() if t]
        for text in texts:
            losses = []
            for m in masks:
                inp = model._prepare_inference_inputs(
                    text=text, num_target_tokens=n_tok, lang="qxp", denoise=False
                )
                ids = inp["input_ids"].clone()
                m_dev = m.to(ids.device)
                target = torch.where(m_dev, torch.full_like(codes, cfg.audio_mask_id), codes)
                ids[0, :, -n_tok:] = target
                labels = torch.full_like(ids, -100)
                labels[0, :, -n_tok:] = torch.where(m_dev, codes, torch.full_like(codes, -100))
                # bidirectional attention, as in OmniVoice's own generation (None = causal)
                full = torch.ones(1, 1, ids.shape[-1], ids.shape[-1], dtype=torch.bool)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    out = model(
                        input_ids=ids,
                        audio_mask=inp["audio_mask"],
                        labels=labels,
                        attention_mask=full.to(ids.device),
                    )
                    lp0 = out.logits[0, 0, -n_tok:].float().log_softmax(-1)
                    nll0 = -lp0[torch.arange(n_tok), codes[0]][m_dev].mean()
                losses.append((float(out.loss), float(nll0)))
            loss, loss0 = np.mean(losses, axis=0)
            rows.append({"clip_id": cid, "text": text, "lp_tts": -loss, "lp_tts0": -loss0})
        if n % 100 == 0:
            print(f"{n}/{man.height} clips ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pl.DataFrame(rows).write_parquet(args.out)
    print(f"{len(rows)} candidates scored -> {args.out}")


if __name__ == "__main__":
    main()
