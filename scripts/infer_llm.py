"""Transcribe manifest roles with an omniASR LLM-ASR model (+ optional LoRA / trainable weights).

Run inside ``envs/omni``:
    uv run python /workspace/simbig/repo/scripts/infer_llm.py --name llm7b_lora_v1 \
        --card omniASR_LLM_7B_v2 --decoder-lora 16 \
        --weights /workspace/simbig/artifacts/runs/llm7b_lora_v1/best_trainable.pt
"""

from __future__ import annotations

import time

import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig

from qxp_asr.audio import load
from qxp_asr.lora import add_lora
from qxp_asr.metrics import wer_cer
from qxp_asr.omni import transcribe_llm
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import pseudo_pool
from qxp_asr.text import normalize

ATTN_FFN = r"\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
DEFAULT_ROLES = "dev_scripted,dev_spont,itest_scripted,itest_spont,ood,test"


def main(
    name: str = typer.Option(...),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    weights: str | None = typer.Option(None, help="trainable-parameter state dict from train_llm"),
    decoder_lora: int = typer.Option(0, help="LoRA rank used on the decoder (0 = none)"),
    encoder_lora: int = typer.Option(0, help="LoRA rank used on the encoder (0 = none)"),
    roles: str = typer.Option(DEFAULT_ROLES),
    pool: bool = typer.Option(False, help="unlabelled spontaneous pool instead of --roles"),
    batch_size: int = typer.Option(8),
    beam: int = typer.Option(1, help="beam size (the official pipeline default is 1 = greedy)"),
    keep_encoder_layers: str = typer.Option(
        "", help="causal ablation: zero the encoder LoRA update outside layers lo:hi"
    ),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    man = pseudo_pool(man) if pool else man.filter(pl.col("role").is_in(roles.split(",")))
    model = load_model(card, device=torch.device("cuda"), dtype=torch.bfloat16)
    for part, rank in (("encoder", encoder_lora), ("llama_decoder", decoder_lora)):
        if rank:
            add_lora(model, f"{part}{ATTN_FFN}", rank=rank, alpha=2 * rank, dropout=0.0)
    if weights:
        state = torch.load(weights, map_location="cuda", weights_only=True)
        missing = set(state) - {n for n, _ in model.named_parameters()}
        if missing:
            msg = f"{len(missing)} weights not in model, e.g. {sorted(missing)[:3]}"
            raise KeyError(msg)
        model.load_state_dict(state, strict=False)
    if keep_encoder_layers:  # causal ablation: the encoder LoRA update only in layers lo..hi-1
        lo, hi = (int(x) for x in keep_encoder_layers.split(":"))
        with torch.no_grad():
            for n, p in model.named_parameters():
                enc_b = n.startswith("encoder.layers.") and "lora_b" in n
                if enc_b and not lo <= int(n.split(".")[2]) < hi:
                    p.zero_()
        print(f"encoder LoRA kept in layers {lo}..{hi - 1} only")
    # keep fp32 LoRA params and fp32 RoPE buffers: autocast handles the mixed precision
    pipe = ASRInferencePipeline(
        model_card=None,
        model=model,
        tokenizer=load_tokenizer(card),
        beam_search_config=Wav2Vec2LlamaBeamSearchConfig(nbest=beam, length_norm=False),
    )
    ordered = man.sort("duration_s")
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hyps = transcribe_llm(pipe, [load(w) for w in ordered["wav"].to_list()], batch_size)
    print(f"{ordered.height} clips, RTFx {ordered['duration_s'].sum() / (time.time() - t0):.0f}")
    pred = ordered.select("clip_id", "role", "unseen_sentence", "text", "test_type").with_columns(
        hyp=pl.Series([normalize(h) for h in hyps]), raw=pl.Series(hyps)
    )
    (ARTIFACTS / "preds").mkdir(parents=True, exist_ok=True)
    pred.write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
    lab = pred.filter(pl.col("text").is_not_null())
    rows = [
        {"role": r, "unseen": u} | wer_cer(g["text"].to_list(), g["hyp"].to_list())
        for (r, u), g in lab.group_by("role", "unseen_sentence", maintain_order=True)
    ]
    if rows:  # unlabelled pools have nothing to score
        with pl.Config(tbl_rows=30):
            print(pl.DataFrame(rows).sort("role", "unseen"))


if __name__ == "__main__":
    typer.run(main)
