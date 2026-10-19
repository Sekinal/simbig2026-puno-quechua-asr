"""Learned verifier (outcome reward model) for ASR candidates: listwise MWER on cross-fit lists.

"Verifier" stage of the Spectrum -> Verifier -> Signal program (LITERATURE §g.12). RL on unlabelled
audio needs a reward that ranks transcripts correctly; our best reference-free rewards order only
~64-70 % of candidate pairs correctly on spontaneous speech (reports/RESULTS.md, reward bench).
LLM RL solves the same problem with learned outcome reward models (GLM-4.5/5, DeepSeek). Here the
verifier is the LLM-ASR itself (audio-conditioned, LoRA from LoRA v2) trained to *discriminate*:

    loss = sum_i softmax(lp_i / tau) * (err_i - mean err)  +  ce_weight * CE(reference)

over the candidate list of a clip (minimum-WER training, Prabhavalkar et al., 2018; discriminative
rescorers RescoreBERT, Xu et al., 2022). The lists come from cross-fit decoding
(``scripts/runs/22_crossfit_serverB.sh``): models that heard neither the speaker nor the sentence,
so their errors look like test-time errors (decoding the training set with the full models gives
memorised, error-free lists). err = WER + 0.5 CER against the gold transcript.

Evaluation (every ``--eval-every`` steps): pick the highest-scoring candidate of the dev pool
(``--select-pool``, e.g. ``pool_dev_in``) per clip; report selection WER and pairwise ranking
accuracy on dev unseen sentences, dev spontaneous and OOD.

Run inside ``envs/omni``:
    uv run python scripts/train_verifier.py --run verifier_v1 \
        --init .../llm7b_lora_v2/best_trainable.pt --cands crossfit0,crossfit1
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model

from qxp_asr.audio import load
from qxp_asr.lora import add_lora, checkpoint_layers
from qxp_asr.metrics import wer_cer
from qxp_asr.nbest import cached_audio, make_batch, sequence_logprobs
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.speed import adamw, enable_fast_math
from qxp_asr.text import normalize
from train_rl import DECODER_LORA, reward


def candidate_lists(
    table: pl.DataFrame, man: pl.DataFrame, max_cands: int
) -> dict[str, list[tuple[str, str, list[str], np.ndarray]]]:
    """Per corpus: (wav, reference, candidate texts, errors) per clip; clips whose candidates all
    have the same error carry no ranking signal and are dropped."""
    lab = man.select("clip_id", "wav", "text", "corpus").drop_nulls()
    out: dict[str, list] = {"scripted": [], "spontaneous": []}
    for (cid,), g in table.join(lab, on="clip_id").group_by("clip_id"):
        ref = normalize(g["text_right"][0])
        texts = sorted({t for t in map(normalize, g["text"].drop_nulls().to_list()) if t})
        if len(texts) < 2:
            continue
        errs = np.array([-reward(ref, t) for t in texts])
        if errs.max() - errs.min() < 1e-9:
            continue
        if len(texts) > max_cands:  # keep the best and a random rest
            rng = np.random.default_rng(int(hashlib.md5(cid.encode()).hexdigest(), 16) % 2**32)
            keep = [int(errs.argmin())]
            rest = [i for i in range(len(texts)) if i != keep[0]]
            keep += list(rng.choice(rest, size=max_cands - 1, replace=False))
            texts, errs = [texts[i] for i in keep], errs[keep]
        out.setdefault(g["corpus"][0], []).append((g["wav"][0], ref, texts, errs))
    return out


def score(model, enc, wav: torch.Tensor, texts: list[str], pad: int) -> torch.Tensor:
    """Sequence log-probabilities of ``texts`` given one clip (encoder computed once)."""
    batch = make_batch([wav] * len(texts), [enc(t).tolist() for t in texts], pad, "cuda")
    with (
        cached_audio(model, batch.source_seqs[:1], batch.source_seq_lens[:1], len(texts)),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        return sequence_logprobs(model, batch)


def main(
    run: str = typer.Option(...),
    init: str = typer.Option(..., help="decoder-LoRA trainable.pt to start from (LoRA v2)"),
    cands: str = typer.Option(..., help="comma-separated cross-fit candidate tables"),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    max_cands: int = typer.Option(8),
    clips: int = typer.Option(4, help="clips per step"),
    spont_frac: float = typer.Option(0.5, help="share of spontaneous clips per step"),
    tau: float = typer.Option(1.0),
    ce_weight: float = typer.Option(0.1),
    lr: float = typer.Option(2e-5),
    steps: int = typer.Option(1500),
    eval_every: int = typer.Option(250),
    select_pool: str = typer.Option("pool_dev_in"),
    eval_clips: int = typer.Option(150),
    seed: int = typer.Option(0),
) -> None:
    cfg = dict(locals())
    torch.manual_seed(seed)
    enable_fast_math()
    out_dir = ARTIFACTS / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    table = pl.concat(
        [pl.read_parquet(ARTIFACTS / "cands" / f"{c}.parquet") for c in cands.split(",")],
        how="diagonal_relaxed",
    ).select("clip_id", "text")
    pools = candidate_lists(table, man, max_cands)
    print({c: len(v) for c, v in pools.items()}, "clips with ranking signal", flush=True)

    # dev selection sets: candidates of the dev pool with gold references
    pool = pl.read_parquet(ARTIFACTS / "cands" / f"{select_pool}.parquet").select("clip_id", "text")
    sets = {
        "dev_unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
        "dev_spont": (pl.col("role") == "dev_spont") & (pl.col("duration_s") <= 39),
        "ood": pl.col("role") == "ood",
    }
    dev = {}
    for name, cond in sets.items():
        sub = man.filter(cond & pl.col("text").is_not_null())
        sub = sub.sample(n=min(eval_clips, sub.height), seed=0)
        dev[name] = [
            (w, normalize(t), sorted(set(pool.filter(pl.col("clip_id") == c)["text"].to_list())))
            for c, w, t in sub.select("clip_id", "wav", "text").iter_rows()
        ]

    tok = load_tokenizer(card)
    enc = tok.create_encoder()
    model = load_model(card, device=torch.device("cuda"), dtype=torch.bfloat16)
    model.requires_grad_(False)
    add_lora(model, DECODER_LORA, rank=16, alpha=32, dropout=0.0)
    checkpoint_layers(model.llama_decoder.layers)
    model.load_state_dict(torch.load(init, map_location="cuda", weights_only=True), strict=False)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad_(True)
    opt = adamw(params, lr=lr, betas=(0.9, 0.98))
    model.eval()
    pad = model.target_vocab_info.pad_idx
    log = (out_dir / "log.jsonl").open("a")

    def run_eval(step: int) -> None:
        rec = {"step": step}
        with torch.no_grad():
            for name, items in dev.items():
                refs, picks, correct, pairs = [], [], 0.0, 0
                for w, ref, texts in items:
                    if not texts:
                        continue
                    lp = score(model, enc, torch.from_numpy(load(w)), texts, pad).float().cpu()
                    lp = lp.numpy()
                    err = np.array([-reward(ref, t) for t in texts])
                    refs.append(ref)
                    picks.append(texts[int(lp.argmax())])
                    better = err[:, None] < err[None, :]
                    correct += float((lp[:, None] > lp[None, :])[better].sum())
                    pairs += int(better.sum())
                rec[f"{name}_sel_wer"] = wer_cer(refs, picks)["wer"]
                rec[f"{name}_pair_acc"] = 100 * correct / max(1, pairs)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    run_eval(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        n_sp = round(clips * spont_frac)
        batch = [pools["spontaneous"][i] for i in rng.integers(0, len(pools["spontaneous"]), n_sp)]
        batch += [
            pools["scripted"][i] for i in rng.integers(0, len(pools["scripted"]), clips - n_sp)
        ]
        opt.zero_grad(set_to_none=True)
        stats = []
        for w, ref, texts, errs in batch:  # one clip at a time: memory ~ one candidate list
            wav = torch.from_numpy(load(w))
            lp = score(model, enc, wav, texts, pad).float()
            post = torch.softmax(lp / tau, dim=0)
            e = torch.tensor(errs, device="cuda", dtype=torch.float)
            mwer = (post * (e - e.mean())).sum()
            ce = -score(model, enc, wav, [ref], pad).float().sum()
            loss = (mwer + ce_weight * ce) / len(batch)
            loss.backward()
            stats.append((float(mwer), float(ce), float(e[post.argmax()] - e.min())))
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        rec = {
            "step": step,
            "mwer": float(np.mean([s[0] for s in stats])),
            "ce": float(np.mean([s[1] for s in stats])),
            "regret": float(np.mean([s[2] for s in stats])),  # err(top-scored) - err(best)
            "gnorm": float(gnorm),
            "min": (time.time() - t0) / 60,
        }
        log.write(json.dumps(rec) + "\n")
        if step % 25 == 0:
            print(json.dumps(rec), flush=True)
        if step % eval_every == 0 or step == steps:
            run_eval(step)
            torch.save(
                {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                out_dir / f"step{step}_trainable.pt",
            )


if __name__ == "__main__":
    typer.run(main)
