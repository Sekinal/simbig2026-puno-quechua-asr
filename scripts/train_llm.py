"""Fine-tune an omniASR LLM-ASR model (wav2vec2 encoder + LLaMA decoder) on Puno Quechua.

Zero-shot omniASR-LLM-7B-v2 is our strongest starting point (dev WER 23.3 scripted / 14.7
spontaneous); most of its scripted error is orthographic (Puno spelling conventions), which
supervised fine-tuning fixes. Per-part training policies keep this within one 48 GB GPU:

* ``--encoder`` / ``--decoder`` ∈ {full, lora, frozen}: full fine-tuning (fp32 master weights),
  LoRA (Hu et al., 2021) on attention + FFN projections, or frozen;
* activation checkpointing of transformer layers (``--grad-ckpt``);
* the model's own SpecAugment-style masking and language-id dropout stay active in train mode;
* data pipeline, sampling weights and augmentation shared with the CTC trainer.

Run inside ``envs/omni``:
    uv run python /workspace/simbig/repo/scripts/train_llm.py --run llm1b_v1 \
        --card omniASR_LLM_1B_v2 --encoder full --decoder lora
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.models.hub import load_model
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
from torch.utils.data import DataLoader

from qxp_asr.audio import load
from qxp_asr.data import AsrDataset, Batch, WeightedDynamicBatchSampler, collate
from qxp_asr.lora import add_lora, checkpoint_layers
from qxp_asr.metrics import wer_cer
from qxp_asr.omni import LANG, transcribe_llm
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import training_rows
from qxp_asr.speed import adamw, enable_fast_math
from qxp_asr.splits import crossfit_excluded, with_crossfit_folds
from qxp_asr.text import normalize
from qxp_asr.textlm import TextSampler, text_lm_loss

ATTN_FFN = r"\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"


def configure(
    model, part: str, policy: str, rank: int, layers: str = ""
) -> list[torch.nn.Parameter]:
    """Apply a training policy to ``encoder`` or ``llama_decoder``; return its trainable params.

    ``layers="lo:hi"`` restricts LoRA to layers lo..hi-1 (the others keep B = 0, i.e. no update,
    so checkpoints load into a full-depth LoRA model unchanged)."""
    module = getattr(model, part)
    if policy == "frozen":
        module.requires_grad_(False)
        return []
    if policy == "full":
        module.float().requires_grad_(True)
        return list(module.parameters())
    if policy == "lora":
        module.requires_grad_(False)
        pattern = f"{part}{ATTN_FFN}"
        if layers:
            lo, hi = (int(x) for x in layers.split(":"))
            idx = "|".join(str(i) for i in range(lo, hi))
            pattern = pattern.replace(r"\.layers\.\d+\.", rf"\.layers\.(?:{idx})\.")
        wrapped = add_lora(model, pattern, rank=rank, alpha=2 * rank)
        print(f"{part}: LoRA on {len(wrapped)} projections")
        return [p for n, p in module.named_parameters() if "lora_" in n]
    msg = f"unknown policy {policy!r}"
    raise ValueError(msg)


def to_seq2seq(batch: Batch, device: str, pad_idx: int) -> Seq2SeqBatch:
    targets = batch.targets.clone()
    for i, n in enumerate(batch.target_lens):
        targets[i, n:] = pad_idx
    return Seq2SeqBatch(
        source_seqs=batch.seqs.to(device, torch.bfloat16),
        source_seq_lens=batch.seq_lens,
        target_seqs=targets.to(device),
        target_seq_lens=batch.target_lens,
        example={"lang": [LANG] * len(batch.seq_lens)},
    )


@torch.inference_mode()
def evaluate(model, tokenizer, dev: dict[str, pl.DataFrame], batch_size: int) -> dict:
    pipe = ASRInferencePipeline(model_card=None, model=model, tokenizer=tokenizer)
    out = {}
    for name, rows in dev.items():
        wavs = [load(w) for w in rows["wav"].to_list()]
        # autocast: fully fine-tuned parts hold fp32 weights, the pipeline feeds bf16 audio
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hyps = transcribe_llm(pipe, wavs, batch_size)
        scores = wer_cer(rows["text"].to_list(), [normalize(h) for h in hyps])
        out[f"{name}_wer"], out[f"{name}_cer"] = scores["wer"], scores["cer"]
    model.train()
    return out


class TwoViewSampler:
    """Repeat every index batch, so each clip appears twice; the dataset augments every item
    independently, so the copies are two different distortions of the same utterance."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def set_epoch(self, epoch: int) -> None:
        self.inner.set_epoch(epoch)

    def __iter__(self):
        for idx in self.inner:
            yield list(idx) + list(idx)

    def __len__(self) -> int:
        return len(self.inner)


def consistency_loss(model, batch: Seq2SeqBatch, beta: float) -> torch.Tensor:
    """Per-token CE on both views + beta x symmetric KL between the two views' token distributions
    (R-Drop / consistency-regularized CTC applied to an LLM-ASR decoder). Rows i and i + B/2 are
    two augmentations of the same clip with the same target."""
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
    mask = (targets != vi.pad_idx).float()
    ce = -(logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()
    half = logp.shape[0] // 2
    a, b = logp[:half], logp[half:]
    kl = (a.exp() * (a - b)).sum(-1) + (b.exp() * (b - a)).sum(-1)  # symmetric, per token
    kl = (0.5 * kl * mask[:half]).sum() / mask[:half].sum()
    return ce + beta * kl


def lr_scale(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def main(
    run: str = typer.Option(...),
    card: str = typer.Option("omniASR_LLM_1B_v2"),
    encoder: str = typer.Option("full"),
    decoder: str = typer.Option("lora"),
    rank: int = typer.Option(32),
    sources: str = typer.Option("scripted:1,spont_gold:3,spont_silver:1"),
    mode: str = typer.Option("dev"),
    steps: int = typer.Option(4000),
    lr_full: float = typer.Option(2e-5),
    lr_lora: float = typer.Option(2e-4),
    extra: str = typer.Option(
        "", help="also train these modules fully: encoder_proj, lang_embeddings, norms, ..."
    ),
    lr_extra: float = typer.Option(5e-5),
    augment: str = typer.Option("mild", help="waveform augmentation profile: mild | strong"),
    encoder_layers: str = typer.Option(
        "", help="LoRA on encoder layers lo:hi only (e.g. 64:128 = top half of the 7B encoder)"
    ),
    consistency: float = typer.Option(
        0.0, help="two augmented views per clip; weight of their symmetric KL (0 = off)"
    ),
    warmup: int = typer.Option(300),
    max_batch_seconds: float = typer.Option(120.0),
    grad_accum: int = typer.Option(2),
    grad_ckpt: bool = typer.Option(
        False, help="also recompute decoder activations (a trained encoder is always recomputed)"
    ),
    eval_every: int = typer.Option(500),
    eval_clips: int = typer.Option(300),
    eval_batch: int = typer.Option(8),
    workers: int = typer.Option(8),
    seed: int = typer.Option(0),
    pseudo: str = typer.Option(""),
    text_corpus: str = typer.Option("", help="decoder text-CPT corpus (one sentence per line)"),
    text_weight: float = typer.Option(0.0, help="weight of the text-only LM loss (0 = off)"),
    text_batch: int = typer.Option(32, help="sentences per text-LM batch"),
    exclude_fold: int = typer.Option(
        -1, help="cross-fit: drop fold-k speakers and fold-k sentences (qxp_asr.splits)"
    ),
) -> None:
    cfg = dict(locals())
    torch.manual_seed(seed)
    enable_fast_math()
    out_dir = ARTIFACTS / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    device = "cuda"

    tokenizer = load_tokenizer(card)
    enc = tokenizer.create_encoder()
    pad_idx = tokenizer.vocab_info.pad_idx
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    weights = {k: float(v) for k, v in (s.split(":") for s in sources.split(","))}
    pseudo_map = {k: Path(v) for k, v in (s.split("=") for s in pseudo.split(",") if s)}
    rows = training_rows(man, list(weights), mode, pseudo_map)
    if exclude_fold >= 0:
        folds = with_crossfit_folds(man).select("clip_id", "spk_fold", "sent_fold", "corpus")
        rows = rows.join(folds, on="clip_id", how="left").filter(
            ~crossfit_excluded(exclude_fold).fill_null(False)
        )
    print(rows.group_by("source").agg(n=pl.len(), hours=pl.col("duration_s").sum() / 3600))
    target_ids = [enc(t).tolist() for t in rows["target"].to_list()]
    train_ds = AsrDataset(
        rows, augment=True, seed=seed, target_ids=target_ids, augment_profile=augment
    )
    sampler = WeightedDynamicBatchSampler(
        train_ds.durations, train_ds.sources, weights, max_batch_seconds, seed=seed
    )
    if consistency > 0:  # two augmented views of every clip in each batch
        sampler = TwoViewSampler(sampler)
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )
    labelled = pl.col("text").is_not_null()
    dev = {
        "dev_unseen": man.filter(
            (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence") & labelled
        ),
        "dev_spont": man.filter((pl.col("role") == "dev_spont") & labelled),
    }
    dev = {
        k: v.sample(n=min(eval_clips, v.height), seed=0).sort("duration_s") for k, v in dev.items()
    }

    model = load_model(card, device=torch.device(device), dtype=torch.bfloat16)
    model.requires_grad_(False)
    full_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []
    for part, policy in (("encoder", encoder), ("llama_decoder", decoder)):
        trainable_part = configure(
            model, part, policy, rank, encoder_layers if part == "encoder" else ""
        )
        (lora_params if policy == "lora" else full_params).extend(trainable_part)
    # Activation checkpointing only where it buys necessary memory: a trained 7B encoder does not
    # fit otherwise; the decoder (frozen encoder: ~20 of 46 GB) runs ~25-30 % faster without it.
    if encoder != "frozen":
        checkpoint_layers(model.encoder.layers)
    if grad_ckpt and decoder != "frozen":
        checkpoint_layers(model.llama_decoder.layers)
    extra_params: list[torch.nn.Parameter] = []
    for name in [x for x in extra.split(",") if x]:  # small modules trained fully (fp32)
        if name == "norms":  # every LayerNorm of the encoder and the decoder
            mods = [
                m
                for n, m in model.named_modules()
                if n.startswith(("encoder.", "llama_decoder.")) and "norm" in n.rsplit(".", 1)[-1]
            ]
        else:
            mods = [model.get_submodule(name)]
        for m in mods:
            m.float().requires_grad_(True)
            extra_params.extend(m.parameters())
        print(f"extra: {name} ({sum(p.numel() for m in mods for p in m.parameters()) / 1e6:.2f} M)")
    groups = []
    if extra_params:
        groups.append({"params": extra_params, "base_lr": lr_extra, "weight_decay": 0.0})
    if full_params:
        groups.append({"params": full_params, "base_lr": lr_full, "weight_decay": 0.01})
    if lora_params:
        groups.append({"params": lora_params, "base_lr": lr_lora, "weight_decay": 0.0})
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_train / 1e6:.1f} M")
    opt = adamw(groups, lr=lr_lora, betas=(0.9, 0.98))
    model.train()

    log = (out_dir / "log.jsonl").open("a")
    best, step, t0 = float("inf"), 0, time.time()
    loss_acc, n_acc, audio_acc, epoch = 0.0, 0, 0.0, 0
    metrics = evaluate(model, tokenizer, dev, eval_batch)
    print(json.dumps({"step": 0} | metrics), flush=True)
    trainable = [p for g in groups for p in g["params"]]
    text = TextSampler(Path(text_corpus), enc, seed=seed) if text_weight > 0 else None
    while step < steps:
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # omniASR returns mean-token CE x batch size; renormalise to a per-token mean so
                # the effective learning rate does not depend on how many clips fit in a batch.
                if consistency > 0:
                    loss = consistency_loss(model, to_seq2seq(batch, device, pad_idx), consistency)
                else:
                    loss = model(to_seq2seq(batch, device, pad_idx)) / len(batch.seq_lens)
                if text is not None:  # decoder text-CPT interleaved with ASR (keeps audio skills)
                    loss = loss + text_weight * text_lm_loss(model, text(text_batch), device)
            (loss / grad_accum).backward()
            loss_acc += float(loss.detach())
            n_acc += 1
            audio_acc += sum(batch.seq_lens) / 16_000
            if (micro + 1) % grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(step, steps, warmup)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                lora_norm = sum(
                    float(p.detach().norm()) for n, p in model.named_parameters() if "lora_b" in n
                )
                rec = {
                    "step": step,
                    "loss": loss_acc / n_acc,
                    "lora_b_norm": lora_norm,
                    "gnorm": float(gnorm),
                    "audio_h": audio_acc / 3600,
                    "elapsed_min": (time.time() - t0) / 60,
                }
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                loss_acc, n_acc = 0.0, 0
            if step % eval_every == 0 or step == steps:
                metrics = evaluate(model, tokenizer, dev, eval_batch)
                score = (metrics["dev_unseen_wer"] + metrics["dev_spont_wer"]) / 2
                rec = {"step": step, "eval": metrics, "score": score}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                state = {
                    n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
                }
                torch.save(state, out_dir / "last_trainable.pt")
                if score < best:
                    best = score
                    torch.save(state, out_dir / "best_trainable.pt")
            if step >= steps:
                break
        epoch += 1
    print(f"done; best score {best:.2f}")


if __name__ == "__main__":
    typer.run(main)
