"""Continued self-supervised pretraining (CPT) of an omniASR wav2vec 2.0 encoder on qxp audio.

Inspirations (docs/LITERATURE.md §c C1 / §g.4): continued pretraining of wav2vec 2.0 on
in-language audio before fine-tuning (Hsu et al., 2021, "Robust wav2vec 2.0", arXiv:2104.01027;
DeHaven & Billa, 2022, arXiv:2207.00659; Nowakowski et al., 2023, arXiv:2301.07295); the
organizers' XLS-R CPT gained 42 % relative (Huaman et al., 2026). omniASR saw qxp only as a sliver
of 4.3M hours, so in-language CPT may still help. LoRA-CPT (Hu et al., 2021) updates low-rank
adapters on the encoder self-attention/FFN projections only, limiting drift from the multilingual
solution; adapters are merged into the encoder at the end so fine-tuning sees a plain encoder.

Objective: the model's own wav2vec 2.0 loss (masked contrastive prediction of quantized targets +
codebook diversity + feature penalty). Audio: every training-role clip (scripted, spontaneous incl.
untranscribed and pause-cut segments) — never dev, itest, OOD or test audio.

Output: ``artifacts/cpt/<run>/encoder.pt`` = state dict of ``encoder_frontend.*`` and ``encoder.*``
(LoRA merged), loadable by ``scripts/train_ctc.py --init-encoder``.

Run inside ``envs/omni``:
    uv run python scripts/cpt_w2v.py --run cpt_w2v1b_lora --lora 64 --steps 8000
"""

from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.models.hub import load_model
from fairseq2.nn.batch_layout import BatchLayout
from torch.utils.data import DataLoader

from qxp_asr.data import AsrDataset, WeightedDynamicBatchSampler, collate
from qxp_asr.lora import LoRA, add_lora, checkpoint_layers
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import roles_for_mode
from qxp_asr.speed import adamw, enable_fast_math

ENCODER_LORA = r"encoder\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(inner|output)_proj)"


def merge_lora(model: torch.nn.Module) -> None:
    """Fold every LoRA update into its base weight and drop the adapters."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRA):
            with torch.no_grad():
                delta = (mod.lora_b @ mod.lora_a) * mod.scale
                mod.base.weight += delta.to(mod.base.weight.dtype)
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, child, mod.base)


def main(
    run: str = typer.Option(...),
    card: str = typer.Option("omniASR_W2V_1B"),
    lora: int = typer.Option(64, help="LoRA rank; 0 = full continued pretraining"),
    steps: int = typer.Option(8000),
    lr: float = typer.Option(2e-4, help="LoRA lr (full CPT: use ~2e-5)"),
    warmup: int = typer.Option(500),
    max_batch_seconds: float = typer.Option(160.0),
    grad_ckpt: bool = typer.Option(True, help="recompute encoder activations (needed on 48 GB)"),
    max_s: float = typer.Option(25.0),
    workers: int = typer.Option(8),
    save_every: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    cfg = dict(locals())
    torch.manual_seed(seed)
    enable_fast_math()
    out_dir = ARTIFACTS / "cpt" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    man = pl.read_parquet(MANIFESTS / "all.parquet")
    rows = man.filter(
        pl.col("role").is_in(roles_for_mode("dev"))
        & pl.col("corpus").is_in(["scripted", "spontaneous"])
        & pl.col("duration_s").is_between(1.0, max_s)
    ).select("clip_id", "wav", "duration_s", target=pl.lit(""), source=pl.col("corpus"))
    hours = rows.group_by("source").agg((pl.col("duration_s").sum() / 3600).round(1))
    print(f"CPT audio: {rows.height} clips; hours by source: {hours.to_dicts()}")
    ds = AsrDataset(rows, augment=True, seed=seed, target_ids=[[1]] * rows.height)
    sampler = WeightedDynamicBatchSampler(
        ds.durations,
        ds.sources,
        {"scripted": 1.0, "spontaneous": 1.0},
        max_batch_seconds,
        seed=seed,
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers, pin_memory=True,
        persistent_workers=True,
    )  # fmt: skip

    model = load_model(card, device=torch.device("cuda"), dtype=torch.float32)
    if lora:
        model.requires_grad_(False)
        wrapped = add_lora(model, ENCODER_LORA, rank=lora, alpha=lora, dropout=0.0)
        params = [p for n, p in model.named_parameters() if "lora_" in n]
        print(
            f"LoRA-CPT on {len(wrapped)} projections, {sum(p.numel() for p in params) / 1e6:.1f} M"
        )
    else:
        model.encoder_frontend.feature_extractor.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
    if grad_ckpt:
        checkpoint_layers(model.encoder.layers)
    opt = adamw(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.train()

    log = (out_dir / "log.jsonl").open("a")
    step, epoch, t0, acc = 0, 0, time.time(), []
    while step < steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            seqs = batch.seqs.to("cuda", non_blocking=True)
            layout = BatchLayout(seqs.shape, seq_lens=batch.seq_lens, device=seqs.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, out = model(seqs, layout)
            # fairseq2 returns sums over masked positions: normalise per masked frame
            n_masked = max(1, out.logits.size(0) * out.logits.size(1))
            total = loss.aggregate / n_masked
            scale = (
                (step + 1) / warmup
                if step < warmup
                else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
            )
            for g in opt.param_groups:
                g["lr"] = lr * scale
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            acc.append(
                (float(loss.contrastive) / n_masked, float(loss.diversity) / n_masked, float(total))
            )
            if step % 50 == 0:
                c, d, t = np.mean(acc, axis=0)
                rec = {"step": step, "contrastive": c, "diversity": d, "total": t,
                       "min": (time.time() - t0) / 60}  # fmt: skip
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                acc = []
            if step % save_every == 0 or step >= steps:
                export(model, out_dir / f"encoder_step{step}.pt", merged=lora > 0)
            if step >= steps:
                break
        epoch += 1
    export(model, out_dir / "encoder.pt", merged=lora > 0)


def export(model: torch.nn.Module, path, merged: bool) -> None:
    """Save encoder_frontend.* + encoder.* weights (LoRA folded in when used)."""
    if merged:  # merge on a CPU copy so training can continue with the adapters
        tmp = copy.deepcopy(model).cpu()
        merge_lora(tmp)
        source = tmp.state_dict()
    else:
        source = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    keep = ("encoder_frontend.", "encoder.")
    torch.save({k: v for k, v in source.items() if k.startswith(keep)}, path)


if __name__ == "__main__":
    typer.run(main)
