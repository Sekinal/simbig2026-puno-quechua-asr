"""Run an omniASR checkpoint (CTC or LLM) over manifest roles; save hypotheses (+ CTC posteriors).

Run inside ``envs/omni``:
    uv run python /workspace/simbig/repo/scripts/omni_infer.py omniASR_CTC_1B_v2 \
        --roles dev_scripted,dev_spont,test --name zs_ctc1b
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer

from qxp_asr.audio import load
from qxp_asr.metrics import wer_cer
from qxp_asr.omni import load_ctc, run_ctc, transcribe_llm
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.text import normalize


def score(df: pl.DataFrame) -> pl.DataFrame:
    lab = df.filter(pl.col("text").is_not_null())
    rows = [
        {"role": role, "unseen": unseen, "hyp": col}
        | wer_cer(g["text"].to_list(), g[col].to_list())
        for (role, unseen), g in lab.group_by("role", "unseen_sentence", maintain_order=True)
        for col in ("hyp", "hyp_norm")
    ]
    return pl.DataFrame(rows).sort("role", "unseen", "hyp")


def main(
    card: str,
    roles: str = typer.Option("dev_scripted,dev_spont,test"),
    name: str = typer.Option(...),
    batch_seconds: float = typer.Option(400.0),
    llm_batch: int = typer.Option(8),
    save_posteriors: bool = typer.Option(True),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("role").is_in(roles.split(",")))
    items = list(man.select("clip_id", "wav", "duration_s").iter_rows())
    tokenizer = load_tokenizer(card)
    decode = tokenizer.create_decoder(skip_special_tokens=True)
    t0 = time.time()
    hyps: dict[str, str] = {}
    if "CTC" in card:
        model = load_ctc(card)
        post_dir = ARTIFACTS / "posteriors" / name
        post_dir.mkdir(parents=True, exist_ok=True)
        for out in run_ctc(model, items, batch_seconds):
            hyps[out.clip_id] = decode(torch.from_numpy(out.greedy_ids))
            if save_posteriors:
                np.save(post_dir / f"{out.clip_id}.npy", out.log_probs)
    else:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        pipe = ASRInferencePipeline(model_card=card)
        ordered = sorted(items, key=lambda x: x[2])
        texts = transcribe_llm(pipe, [load(w) for _, w, _ in ordered], llm_batch)
        hyps.update({clip_id: t for (clip_id, _, _), t in zip(ordered, texts, strict=True)})
    elapsed = time.time() - t0
    audio_h = man["duration_s"].sum() / 3600
    rtfx = audio_h * 3600 / elapsed
    print(f"{len(hyps)} clips, {audio_h:.2f} h audio in {elapsed:.0f}s (RTFx {rtfx:.0f})")

    pred = (
        man.select("clip_id", "role", "corpus", "unseen_sentence", "text", "test_type")
        .join(pl.DataFrame({"clip_id": list(hyps), "hyp": list(hyps.values())}), on="clip_id")
        .with_columns(hyp_norm=pl.col("hyp").map_elements(normalize, return_dtype=pl.String))
    )
    out_dir = ARTIFACTS / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(out_dir / f"{name}.parquet")
    with pl.Config(tbl_rows=50, tbl_width_chars=200):
        print(score(pred))


if __name__ == "__main__":
    typer.run(main)
