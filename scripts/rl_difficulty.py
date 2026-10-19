"""Difficulty pre-pass for RL (curriculum / online-difficulty filtering, LITERATURE §g.12).

GRPO-style updates learn only from groups whose rewards differ. LLM RL recipes (GLM-4.5, Kimi
k1.5, DAPO's dynamic sampling) therefore filter prompts by difficulty: skip those the policy always
solves or never solves. For ASR we decode every candidate RL clip once with beam ``--beam``
under the initial adapter and weight it by

    w = err(top-1)   if a better hypothesis exists in the beam, else 0

(err = WER + 0.5 CER against the gold text): clips where the model errs but its own beam already
contains a better answer are exactly where a policy-gradient step has signal.
``train_rl.py --difficulty`` samples read clips by w with a uniform floor.

Run inside ``envs/omni``:
    uv run python scripts/rl_difficulty.py --init .../llm7b_lora_v2/best_trainable.pt \
        --sentence-half 0 --out rl_difficulty_h0
"""

from __future__ import annotations

import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model

from qxp_asr.audio import load
from qxp_asr.lora import add_lora
from qxp_asr.nbest import NBestBeamSearch, make_batch
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import training_rows
from qxp_asr.text import normalize
from train_rl import DECODER_LORA, half_of, reward


def main(
    init: str = typer.Option(...),
    sentence_half: int = typer.Option(0),
    out: str = typer.Option(...),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    beam: int = typer.Option(8),
    batch: int = typer.Option(8),
    max_s: float = typer.Option(20.0),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").with_columns(
        half=pl.col("sentence_id").map_elements(
            lambda x: half_of(x) if x else -1, return_dtype=pl.Int64
        )
    )
    held = man.filter((pl.col("role") == "holdout_sent") & (pl.col("half") == sentence_half))
    rows = training_rows(held, ["scripted"], "org").filter(pl.col("duration_s") <= max_s)
    tok = load_tokenizer(card)
    dec = tok.create_decoder(skip_special_tokens=True)
    model = load_model(card, device=torch.device("cuda"), dtype=torch.bfloat16)
    add_lora(model, DECODER_LORA, rank=16, alpha=32, dropout=0.0)
    model.load_state_dict(torch.load(init, map_location="cuda", weights_only=True), strict=False)
    model.eval()
    gen = NBestBeamSearch(model, beam)
    pad = model.target_vocab_info.pad_idx
    recs = []
    for i in range(0, rows.height, batch):
        part = rows.slice(i, batch)
        wavs = [torch.from_numpy(load(w)) for w in part["wav"].to_list()]
        src = make_batch(wavs, [[0]] * len(wavs), pad, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            groups, _ = gen.generate_nbest(src)
        for cid, ref, g in zip(part["clip_id"], part["target"], groups, strict=True):
            errs = [-reward(ref, normalize(dec(torch.tensor(h)))) for h in g]
            top, best = errs[0], min(errs)
            recs.append(
                {"clip_id": cid, "top1_err": top, "best_err": best, "w": top * (best < top)}
            )
    res = pl.DataFrame(recs)
    res.write_parquet(ARTIFACTS / "runs" / f"{out}.parquet")
    print(
        f"{res.height} clips; with signal {(res['w'] > 0).sum()};"
        f" mean top-1 err {res['top1_err'].mean():.3f}, beam oracle {res['best_err'].mean():.3f}"
    )


if __name__ == "__main__":
    typer.run(main)
