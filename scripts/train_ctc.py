"""Fine-tune an omniASR CTC checkpoint on Puno Quechua (single GPU, bf16 autocast).

Recipe (see docs/LITERATURE.md):
* init from ``omniASR_CTC_*_v2`` (Omnilingual ASR, arXiv:2511.09690 §5.7.5: CTC-seeded
  single-language fine-tuning is the strongest small-model recipe for qxp), output head pruned to
  our alphabet;
* model-internal time/channel masking (SpecAugment, Park et al. 2019) and layer drop stay active;
* frozen CNN feature encoder (wav2vec 2.0 fine-tuning practice, Baevski et al. 2020);
* AdamW + tri-stage LR schedule (warmup / hold / exponential decay, as in fairseq w2v2 fine-tuning);
* weighted multi-source sampling + trim/speed/noise augmentation (``qxp_asr.data``/``augment``);
* greedy dev WER every ``eval_every`` steps; best checkpoint by the mean of unseen-sentence
  scripted WER and spontaneous WER (the test reads new sentences — see qxp_asr.splits).

Run inside ``envs/omni``:
    uv run python /workspace/simbig/repo/scripts/train_ctc.py --run ctc1b_v1 \
        --sources scripted:1,spont_gold:3,spont_silver:1
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.nn.batch_layout import BatchLayout
from torch.utils.data import DataLoader

from qxp_asr.data import AsrDataset, Batch, WeightedDynamicBatchSampler, collate
from qxp_asr.metrics import wer_cer
from qxp_asr.omni import load_qxp_ctc
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import training_rows
from qxp_asr.speed import adamw, enable_fast_math
from qxp_asr.vocab import ctc_collapse, decode


@dataclass
class Config:
    run: str
    card: str
    sources: dict[str, float]
    mode: str
    steps: int
    lr: float
    head_lr_mult: float
    warmup_frac: float
    hold_frac: float
    final_lr_scale: float
    max_batch_seconds: float
    grad_accum: int
    weight_decay: float
    max_grad_norm: float
    eval_every: int
    seed: int
    workers: int
    init_checkpoint: str | None
    pseudo: dict[str, str]


def tri_stage(step: int, cfg: Config) -> float:
    warm = int(cfg.warmup_frac * cfg.steps)
    hold = int(cfg.hold_frac * cfg.steps)
    if step < warm:
        return 0.01 + 0.99 * step / max(1, warm)
    if step < warm + hold:
        return 1.0
    frac = (step - warm - hold) / max(1, cfg.steps - warm - hold)
    return math.exp(math.log(cfg.final_lr_scale) * frac)


def forward_loss(model, batch: Batch, device: str) -> tuple[torch.Tensor, int]:
    seqs = batch.seqs.to(device, non_blocking=True)
    layout = BatchLayout(seqs.shape, seq_lens=batch.seq_lens, device=seqs.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, out_layout = model(seqs, layout)
    log_probs = torch.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    loss = torch.nn.functional.ctc_loss(
        log_probs,
        batch.targets.to(device),
        input_lengths=out_layout.seq_lens_pt,
        target_lengths=torch.tensor(batch.target_lens, device=device),
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )
    return loss, sum(batch.target_lens)


@torch.inference_mode()
def evaluate(model, loaders: dict[str, DataLoader], device: str) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, loader in loaders.items():
        refs, hyps = [], []
        for batch in loader:
            seqs = batch.seqs.to(device)
            layout = BatchLayout(seqs.shape, seq_lens=batch.seq_lens, device=seqs.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, out_layout = model(seqs, layout)
            ids = logits.argmax(-1).cpu().numpy()
            for i, n in enumerate(out_layout.seq_lens):
                hyps.append(decode(ctc_collapse(ids[i, :n].tolist())))
            refs.extend(loader.dataset.texts[j] for j in batch_indices(loader, batch))
        scores = wer_cer(refs, hyps)
        out[f"{name}_wer"], out[f"{name}_cer"] = scores["wer"], scores["cer"]
    model.train()
    return out


def batch_indices(loader: DataLoader, batch: Batch) -> list[int]:
    lookup = loader.dataset.index_of
    return [lookup[c] for c in batch.clip_ids]


def eval_loader(
    man: pl.DataFrame, role: str, workers: int, unseen_only: bool = False
) -> DataLoader:
    keep = (pl.col("role") == role) & pl.col("text").is_not_null()
    if unseen_only:
        keep &= pl.col("unseen_sentence")
    rows = man.filter(keep).select(
        "clip_id", "wav", "duration_s", target=pl.col("text"), source=pl.lit(role)
    )
    ds = AsrDataset(rows, augment=False)
    ds.index_of = {c: i for i, c in enumerate(ds.clip_ids)}
    order = np.argsort(ds.durations)[::-1]
    batches, cur, longest = [], [], 0.0
    for i in order:
        d = float(ds.durations[i])
        longest = max(longest, d) if cur else d
        if cur and longest * (len(cur) + 1) > 160:
            batches.append(cur)
            cur, longest = [], d
        cur.append(int(i))
    batches.append(cur)
    return DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=workers)


def parse_sources(spec: str) -> dict[str, float]:
    return {k: float(v) for k, v in (item.split(":") for item in spec.split(","))}


def main(
    run: str = typer.Option(...),
    card: str = typer.Option("omniASR_CTC_1B_v2"),
    sources: str = typer.Option("scripted:1,spont_gold:3,spont_silver:1"),
    mode: str = typer.Option("dev"),
    steps: int = typer.Option(6000),
    lr: float = typer.Option(3e-5),
    head_lr_mult: float = typer.Option(10.0),
    max_batch_seconds: float = typer.Option(200.0),
    grad_accum: int = typer.Option(1),
    eval_every: int = typer.Option(500),
    seed: int = typer.Option(0),
    workers: int = typer.Option(10),
    init_checkpoint: str | None = typer.Option(None),
    freeze_frac: float = typer.Option(0.0, help="freeze this fraction of the lowest layers"),
    pseudo: str = typer.Option("", help="name=path.parquet,... pseudo-label sources"),
    keep_from: float = typer.Option(0.6, help="keep every eval checkpoint after this fraction"),
    init_encoder: str | None = typer.Option(
        None, help="encoder_frontend/encoder weights from scripts/cpt_w2v.py (CTC head kept)"
    ),
) -> None:
    cfg = Config(
        run=run,
        card=card,
        sources=parse_sources(sources),
        mode=mode,
        steps=steps,
        lr=lr,
        head_lr_mult=head_lr_mult,
        warmup_frac=0.08,
        hold_frac=0.32,
        final_lr_scale=0.05,
        max_batch_seconds=max_batch_seconds,
        grad_accum=grad_accum,
        weight_decay=0.01,
        max_grad_norm=5.0,
        eval_every=eval_every,
        seed=seed,
        workers=workers,
        init_checkpoint=init_checkpoint,
        pseudo=dict(item.split("=") for item in pseudo.split(",") if item),
    )
    torch.manual_seed(seed)
    enable_fast_math()
    out_dir = ARTIFACTS / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    device = "cuda"

    man = pl.read_parquet(MANIFESTS / "all.parquet")
    rows = training_rows(
        man, list(cfg.sources), cfg.mode, {k: Path(v) for k, v in cfg.pseudo.items()}
    )
    print(rows.group_by("source").agg(n=pl.len(), hours=pl.col("duration_s").sum() / 3600))
    train_ds = AsrDataset(rows, augment=True, seed=seed)
    sampler = WeightedDynamicBatchSampler(
        train_ds.durations, train_ds.sources, cfg.sources, cfg.max_batch_seconds, seed=seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    dev = {
        "dev_scripted": eval_loader(man, "dev_scripted", 4),
        "dev_unseen": eval_loader(man, "dev_scripted", 4, unseen_only=True),
        "dev_spont": eval_loader(man, "dev_spont", 4),
    }

    model = load_qxp_ctc(card, cfg.init_checkpoint, device, dtype=torch.float32)
    if init_encoder:  # continued-pretrained encoder + pretrained CTC character head
        enc_state = torch.load(init_encoder, map_location="cpu", weights_only=True)
        own = model.state_dict()
        matched = {k: v for k, v in enc_state.items() if k in own and own[k].shape == v.shape}
        expected = [k for k in own if k.startswith(("encoder_frontend.", "encoder."))]
        print(f"init-encoder: {len(matched)}/{len(expected)} encoder tensors loaded")
        assert len(matched) >= 0.95 * len(expected), "encoder checkpoint does not match the model"
        model.load_state_dict(matched, strict=False)
        del enc_state, matched, own
        torch.cuda.empty_cache()
    model.train()
    for p in model.encoder_frontend.feature_extractor.parameters():
        p.requires_grad_(False)
    if freeze_frac > 0:  # large models: fp32 AdamW state of all layers does not fit in 48 GB
        layers = model.encoder.layers
        n_frozen = int(len(layers) * freeze_frac)
        for layer in list(layers)[:n_frozen]:
            for p in layer.parameters():
                p.requires_grad_(False)
        print(f"froze the lowest {n_frozen}/{len(layers)} encoder layers")
    head = list(model.final_proj.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = adamw(
        [{"params": body, "lr_mult": 1.0}, {"params": head, "lr_mult": cfg.head_lr_mult}],
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    log = (out_dir / "log.jsonl").open("a")
    best = float("inf")
    step, epoch, t0 = 0, 0, time.time()
    tokens_acc, loss_acc, audio_acc = 0, 0.0, 0.0
    metrics = evaluate(model, dev, device)
    print(json.dumps({"step": 0} | metrics))
    while step < cfg.steps:
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(train_loader):
            loss, n_tok = forward_loss(model, batch, device)
            (loss / max(1, n_tok) / cfg.grad_accum).backward()
            loss_acc += float(loss.detach())
            tokens_acc += n_tok
            audio_acc += sum(batch.seq_lens) / 16_000
            if (micro + 1) % cfg.grad_accum:
                continue
            scale = tri_stage(step, cfg)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * scale * g["lr_mult"]
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if torch.isfinite(gnorm):
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                rec = {
                    "step": step,
                    "loss_per_tok": loss_acc / max(1, tokens_acc),
                    "gnorm": float(gnorm),
                    "lr": cfg.lr * scale,
                    "audio_h": audio_acc / 3600,
                    "elapsed_min": (time.time() - t0) / 60,
                }
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                tokens_acc, loss_acc = 0, 0.0
            if step % cfg.eval_every == 0 or step == cfg.steps:
                metrics = evaluate(model, dev, device)
                score = (metrics["dev_unseen_wer"] + metrics["dev_spont_wer"]) / 2
                rec = {"step": step, "eval": metrics, "score": score}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if score < best:
                    best = score
                    torch.save(model.state_dict(), out_dir / "best.pt")
                torch.save(model.state_dict(), out_dir / "last.pt")
                if step >= keep_from * cfg.steps:  # ingredients for checkpoint soups
                    torch.save(model.state_dict(), out_dir / f"step{step}.pt")
            if step >= cfg.steps:
                break
        epoch += 1
    print(f"done; best score {best:.2f}")


if __name__ == "__main__":
    typer.run(main)
