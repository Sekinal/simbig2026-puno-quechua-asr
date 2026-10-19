"""Does "adapt the encoder, not the decoder" transfer to another model family? Whisper + LoRA.

The same three-way comparison as for omniASR-LLM (exp34b/44b; LoRA v2): rank-16 LoRA (alpha 32,
``qxp_asr.lora``) on the encoder only, on the decoder only, or on both, trained on exactly the same
rows, source weights, pseudo-labels and mild augmentation (``qxp_asr.sources.training_rows``,
``AsrDataset``); clips > 30 s are dropped (Whisper's window). Whisper has no Quechua token; the
Spanish one is used, as is common for unsupported languages. Encoder LoRA covers self-attention and
MLP; decoder LoRA covers self-attention, cross-attention and MLP. Learning rates as for omniASR
(decoder-only 3e-4, with encoder 2e-4), 1,500 updates of 32 clips, cosine schedule. The last
checkpoint is evaluated (no selection) by greedy decoding on every evaluation set (clips <= 30 s);
predictions go to ``artifacts/preds/<run>.parquet`` for ``scripts/compare_runs.py``.

Runs in ``envs/bench_hf`` with ``PYTHONPATH=src``:
    python scripts/train_whisper.py --run whisper_lora_enc --encoder lora --decoder frozen
    python scripts/train_whisper.py --run whisper_zs --steps 0   # zero-shot reference
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from qxp_asr.audio import load
from qxp_asr.data import AsrDataset, WeightedDynamicBatchSampler
from qxp_asr.lora import add_lora
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import training_rows
from qxp_asr.text import normalize

ENCODER = r"model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|fc1|fc2)"
DECODER = r"model\.decoder\.layers\.\d+\.((self_attn|encoder_attn)\.(q|k|v|out)_proj|fc1|fc2)"
EVAL = {
    "dev_unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
    "itest_unseen": (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"),
    "dev_spont": pl.col("role") == "dev_spont",
    "itest_spont": pl.col("role") == "itest_spont",
    "ood": pl.col("role") == "ood",
}


def lr_scale(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


class FixedBatchSampler:
    """Weighted per-epoch resampling (as for omniASR) cut into fixed-size clip batches."""

    def __init__(self, inner: WeightedDynamicBatchSampler, batch: int, seed: int) -> None:
        self.inner, self.batch, self.rng = inner, batch, np.random.default_rng(seed)

    def set_epoch(self, epoch: int) -> None:
        self.inner.set_epoch(epoch)

    def __iter__(self):
        idx = [i for b in self.inner for i in b]
        self.rng.shuffle(idx)
        for k in range(0, len(idx) - self.batch + 1, self.batch):
            yield idx[k : k + self.batch]


def main(
    run: str = typer.Option(...),
    model_name: str = typer.Option("openai/whisper-large-v3"),
    encoder: str = typer.Option("frozen", help="lora | frozen"),
    decoder: str = typer.Option("frozen", help="lora | frozen"),
    rank: int = typer.Option(16),
    lr: float = typer.Option(0.0, help="0: 3e-4 decoder-only, 2e-4 with the encoder"),
    steps: int = typer.Option(1500),
    warmup: int = typer.Option(100),
    batch: int = typer.Option(16),
    grad_accum: int = typer.Option(2),
    sources: str = typer.Option("scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1"),
    pseudo: str = typer.Option(""),
    mode: str = typer.Option("dev"),
    language: str = typer.Option("es"),
    workers: int = typer.Option(8),
    eval_batch: int = typer.Option(16),
    seed: int = typer.Option(0),
) -> None:
    torch.manual_seed(seed)
    device = "cuda"
    out_dir = ARTIFACTS / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(dict(locals()), default=str, indent=2))
    proc = WhisperProcessor.from_pretrained(model_name)
    proc.tokenizer.set_prefix_tokens(language=language, task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.requires_grad_(False)
    params = []
    for part, policy, pattern in (("encoder", encoder, ENCODER), ("decoder", decoder, DECODER)):
        if policy == "lora":
            names = add_lora(model, pattern, rank=rank, alpha=2 * rank)
            print(f"{part}: LoRA on {len(names)} projections")
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(device)
            p.requires_grad_(True)
            params.append(p)
    lr = lr or (2e-4 if encoder == "lora" else 3e-4)
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f} M, lr {lr}")
    if encoder == "lora":  # 32 x 30 s through the encoder: recompute activations
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prompt_len = len(proc.tokenizer("").input_ids) - 1  # prefix special tokens, without eos

    man = pl.read_parquet(MANIFESTS / "all.parquet")
    if steps > 0:
        weights = {k: float(v) for k, v in (s.split(":") for s in sources.split(","))}
        pseudo_map = {k: Path(v) for k, v in (s.split("=") for s in pseudo.split(",") if s)}
        rows = training_rows(man, list(weights), mode, pseudo_map)
        rows = rows.filter(pl.col("duration_s") <= 30)
        labels = [proc.tokenizer(t).input_ids for t in rows["target"].to_list()]
        ds = AsrDataset(rows, augment=True, seed=seed, target_ids=labels)
        inner = WeightedDynamicBatchSampler(ds.durations, ds.sources, weights, 120.0, seed=seed)
        sampler = FixedBatchSampler(inner, batch, seed)

        def collate(items):
            _, xs, ys = zip(*items, strict=True)
            feats = proc.feature_extractor(
                [x.numpy() for x in xs], sampling_rate=16_000, return_tensors="pt"
            ).input_features
            n = max(len(y) for y in ys)
            lab = torch.full((len(ys), n), -100, dtype=torch.long)
            for i, y in enumerate(ys):
                lab[i, : len(y)] = y
            if (lab[:, 0] == model.config.decoder_start_token_id).all():
                lab = lab[:, 1:]  # the model prepends the start token itself
            return feats, lab

        loader = DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
            persistent_workers=True,
        )  # fmt: skip
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
        model.train()
        step, epoch, micro, t0, acc = 0, 0, 0, time.time(), []
        log = (out_dir / "log.jsonl").open("a")
        while step < steps:
            sampler.set_epoch(epoch)
            for feats, lab in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(
                        input_features=feats.to(device, torch.bfloat16), labels=lab.to(device)
                    )
                (out.loss / grad_accum).backward()
                acc.append(float(out.loss.detach()))
                micro += 1
                if micro % grad_accum:
                    continue
                for g in opt.param_groups:
                    g["lr"] = lr * lr_scale(step, steps, warmup)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    rec = {
                        "step": step,
                        "loss": float(np.mean(acc)),
                        "min": (time.time() - t0) / 60,
                    }
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    acc = []
                if step >= steps:
                    break
            epoch += 1
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                   out_dir / "last_trainable.pt")  # fmt: skip

    model.eval()
    labelled = man.filter(pl.col("text").is_not_null() & (pl.col("duration_s") <= 30))
    rows_out, results = [], []
    for name, cond in EVAL.items():
        d = labelled.filter(cond).sort("duration_s")
        hyps = []
        for k in range(0, d.height, eval_batch):
            part = d.slice(k, eval_batch)
            feats = proc.feature_extractor(
                [load(w) for w in part["wav"].to_list()], sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(device, torch.bfloat16)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(
                    input_features=feats, language=language, task="transcribe", num_beams=1,
                    max_new_tokens=440 - prompt_len,
                )  # fmt: skip
            hyps += [normalize(h) for h in proc.batch_decode(gen, skip_special_tokens=True)]
        score = wer_cer(d["text"].to_list(), hyps)
        results.append(
            {"set": name, "clips": d.height} | {k: round(score[k], 2) for k in ("wer", "cer")}
        )
        rows_out += [
            {"clip_id": c, "role": r, "unseen_sentence": u, "text": t, "hyp": h}
            for (c, r, u, t), h in zip(
                d.select("clip_id", "role", "unseen_sentence", "text").iter_rows(),
                hyps,
                strict=True,
            )
        ]
    pl.DataFrame(rows_out).write_parquet(ARTIFACTS / "preds" / f"{run}.parquet")
    print(pl.DataFrame(results))
    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
