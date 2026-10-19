"""Sequence-level fine-tuning of the LoRA-adapted omniASR-LLM-7B: GSPO or MWER on N-best groups.

The token-level cross-entropy we trained with is not the metric we are scored on. Following
minimum-WER training (Prabhavalkar et al., 2018, ICASSP, "Minimum Word Error Rate Training for
Attention-based Sequence-to-Sequence Models") and group-relative policy optimisation with
sequence-level importance ratios (GSPO: Zheng et al., 2025, arXiv:2507.18071; GRPO: Shao et al.,
2024, arXiv:2402.03300), each step:

1. draws B gold-labelled clips and computes the frozen 7B encoder ONCE per clip;
2. decodes a group of G hypotheses per clip with beam search (``qxp_asr.nbest``);
3. rewards each hypothesis with -(WER + 0.5 CER) against the reference;
4. drops groups without signal or with a likely-misread reference (GLM-5 report,
   arXiv:2602.15763, "dropping off-policy and noisy samples");
5. updates the decoder LoRA with either
   * ``gspo``: group-normalised advantages, length-normalised sequence log-ratio and, by default,
     the GLM-5 / IcePop-style double-sided trust region that *masks* out-of-range sequences
     instead of clipping them (beam-search groups are off-policy; masking is safer than clipping),
     ``inner`` optimisation epochs per group; or
   * ``drgrpo``: on-policy token-sum policy gradient with mean-centred rewards (Dr. GRPO);
   * ``mwer``: expected (mean-centred) error under the renormalised N-best posterior;
   plus ``ce_weight`` x cross-entropy on the reference as an anchor (keeps the policy close to SFT).

Only gold labels are used as rewards (pseudo labels would reinforce their own errors).

``--sentence-half h`` (sentence-cross-fit protocol): the RL clips are training speakers reading
held-out sentences (role ``holdout_sent``) of half ``h`` of the 400 held-out sentences, i.e. gold
clips the SFT model has never seen the text of, so rewards vary; evaluation uses dev + itest clips
of the *other* half (unseen speakers and unseen sentences). ``--objective sft`` trains plain
cross-entropy on the same clips: the control that isolates what RL adds.

Run inside ``envs/omni``:
    uv run python scripts/train_rl.py --run rl_gspo_v1 --init .../llm7b_lora_v2/best_trainable.pt
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time

import kenlm
import numpy as np
import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from rapidfuzz.distance import Levenshtein

from qxp_asr.audio import load
from qxp_asr.augment import VIEWS, view
from qxp_asr.decode import ctc_loglik, load_posteriors
from qxp_asr.lora import add_lora, checkpoint_layers
from qxp_asr.metrics import wer_cer
from qxp_asr.nbest import NBestBeamSearch, cached_audio, make_batch, sequence_logprobs
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import pseudo_pool, training_rows
from qxp_asr.speed import adamw, enable_fast_math
from qxp_asr.text import normalize

ENCODER_LORA = (
    r"encoder\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
)
DECODER_LORA = (
    r"llama_decoder\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
)


OLD = ["fifties", "sixties", "seventies", "eighties", "nineties"]


def verifier_reward(lp: np.ndarray, hyp: str, lm10: kenlm.Model | None) -> float:
    """Reference-free reward: per-character CTC log-likelihood (+ 0.3 x per-char char-LM)."""
    n = max(1, len(hyp))
    r = ctc_loglik(lp, hyp) / n
    if lm10 is not None and hyp:
        r += 0.3 * lm10.score(" ".join("|" if c == " " else c for c in hyp)) * math.log(10) / n
    return max(r, -50.0)


def reward(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    wer = Levenshtein.distance(r, h) / max(1, len(r))
    cer = Levenshtein.distance(ref, hyp) / max(1, len(ref))
    return -(wer + 0.5 * cer)


def half_of(sentence_id: str) -> int:
    """Stable 2-way split of sentences (md5, independent of Python's hash seed)."""
    return int(hashlib.md5(sentence_id.encode()).hexdigest(), 16) % 2


def evaluate(model, tok_dec, rows: pl.DataFrame, nbest: int, device: str) -> dict:
    gen = NBestBeamSearch(model, nbest)
    hyps = []
    for i in range(0, rows.height, 8):
        part = rows.slice(i, 8)
        wavs = [torch.from_numpy(load(w)) for w in part["wav"].to_list()]
        batch = make_batch(wavs, [[0]] * len(wavs), model.target_vocab_info.pad_idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            groups, _ = gen.generate_nbest(batch)
        hyps += [normalize(tok_dec(torch.tensor(g[0]))) for g in groups]
    return wer_cer(rows["text"].to_list(), hyps)


def main(
    run: str = typer.Option(...),
    init: str = typer.Option(..., help="best_trainable.pt of a decoder-LoRA SFT run"),
    encoder_rank: int = typer.Option(0, help="encoder LoRA rank of an enc+dec init (kept frozen)"),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    rank: int = typer.Option(16),
    objective: str = typer.Option(
        "gspo", help="gspo | drgrpo | mwer | sft (cross-entropy control)"
    ),
    sentence_half: int = typer.Option(
        -1, help="0/1: RL on held-out-sentence clips of this half, evaluate on the other half"
    ),
    mode: str = typer.Option("dev"),
    group: int = typer.Option(4),
    clips: int = typer.Option(8, help="clips per step"),
    inner: int = typer.Option(2, help="GSPO optimisation epochs per sampled group"),
    clip_eps: float = typer.Option(0.2),
    update: str = typer.Option(
        "mask", help="mask: drop out-of-trust-region sequences (GLM-5/IcePop); clip: PPO clipping"
    ),
    max_best_err: float = typer.Option(
        0.5, help="drop groups whose best hypothesis still has WER+0.5CER above this (noisy label)"
    ),
    ce_weight: float = typer.Option(0.1),
    lr: float = typer.Option(2e-5),
    steps: int = typer.Option(300),
    max_s: float = typer.Option(20.0),
    eval_every: int = typer.Option(50),
    eval_clips: int = typer.Option(150),
    eval_beam: int = typer.Option(1, help="1 = greedy, as the official pipeline decodes"),
    unlabeled: str = typer.Option(
        "", help="CTC posteriors of the unlabelled pool: enables verifier-reward RL on it"
    ),
    unlabeled_frac: float = typer.Option(0.5, help="share of clips per step from the pool"),
    char_lm: str = typer.Option("", help="char 10-gram for the verifier reward (optional)"),
    seed: int = typer.Option(0),
    grad_ckpt: bool = typer.Option(True, help="recompute decoder activations in backward"),
    scripted_frac: float = typer.Option(
        0.5, help="share of gold clips per step that are read speech"
    ),
    difficulty: str = typer.Option(
        "", help="parquet (clip_id, w) from scripts/rl_difficulty.py: sample read clips by w"
    ),
    difficulty_floor: float = typer.Option(0.2, help="share of uniform sampling kept"),
    rollouts: str = typer.Option(
        "",
        help="mixed rollout sources, e.g. 'beam:2,sample:3,aug:3' (sizes sum to --group); "
        "default: beam-search groups",
    ),
) -> None:
    cfg = dict(locals())
    random.seed(seed)
    torch.manual_seed(seed)
    enable_fast_math()
    device = "cuda"
    out_dir = ARTIFACTS / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    tokenizer = load_tokenizer(card)
    enc, dec = tokenizer.create_encoder(), tokenizer.create_decoder(skip_special_tokens=True)
    man = pl.read_parquet(MANIFESTS / "all.parquet").with_columns(
        half=pl.col("sentence_id").map_elements(
            lambda x: half_of(x) if x else -1, return_dtype=pl.Int64
        )
    )
    rows = training_rows(man, ["scripted", "spont_gold"], mode).filter(
        pl.col("duration_s") <= max_s
    )
    if sentence_half >= 0:  # scripted RL clips: held-out sentences of one half only
        held = man.filter((pl.col("role") == "holdout_sent") & (pl.col("half") == sentence_half))
        held = training_rows(held, ["scripted"], "org").filter(pl.col("duration_s") <= max_s)
        rows = pl.concat([rows.filter(pl.col("source") == "spont_gold"), held])
    # balance read and spontaneous speech in every step (the test is ~half spontaneous audio)
    pools = {s: rows.filter(pl.col("source") == s) for s in ("scripted", "spont_gold")}
    sample_w = None
    if difficulty:  # weights for the read-speech pool; clips without signal get w = 0
        w = pl.read_parquet(difficulty).select("clip_id", "w")
        pools["scripted"] = pools["scripted"].join(w, on="clip_id", how="left").fill_null(0.0)
        raw = pools["scripted"]["w"].to_numpy()
        sample_w = raw / raw.sum()
        print(f"difficulty: {(raw > 0).sum()}/{raw.size} read clips with signal")
    dev = {
        "dev_unseen": man.filter((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence")),
        "dev_spont": man.filter((pl.col("role") == "dev_spont") & (pl.col("duration_s") <= 39)),
    }
    dev = {k: v.sample(n=min(eval_clips, v.height), seed=0) for k, v in dev.items()}
    if sentence_half >= 0:  # all dev + itest clips of the other half's sentences, and OOD
        dev["unseen_other_half"] = man.filter(
            pl.col("role").is_in(["dev_scripted", "itest_scripted"])
            & pl.col("unseen_sentence")
            & (pl.col("half") == 1 - sentence_half)
        )
        dev["ood"] = man.filter(pl.col("role") == "ood")
        del dev["dev_unseen"]
        print({k: v.height for k, v in dev.items()}, f"RL clips: {rows.height}")

    # Hybrid reward (GLM-5 report §3.4): rule-based reward on gold clips; on untranscribed clips a
    # verifier reward from an independent model (our CTC: exact per-character log-likelihood,
    # + char-LM), so RL can also learn from the ~30 h unlabelled (elderly-rich) spontaneous pool.
    upool, lm10 = None, None
    if unlabeled:
        upool = pseudo_pool(man, mode).filter(pl.col("duration_s") <= max_s)
        have = {p.stem for p in (ARTIFACTS / "posteriors" / unlabeled).glob("*.npy")}
        upool = upool.filter(pl.col("clip_id").is_in(list(have))).with_columns(
            w=pl.when(pl.col("age").is_in(OLD)).then(3.0).otherwise(1.0)
        )
        lm10 = kenlm.Model(char_lm) if char_lm else None
        print(f"unlabelled pool: {upool.height} clips ({upool['duration_s'].sum() / 3600:.1f} h)")
    model = load_model(card, device=torch.device(device), dtype=torch.bfloat16)
    model.requires_grad_(False)
    add_lora(model, DECODER_LORA, rank=rank, alpha=2 * rank, dropout=0.0)
    if encoder_rank:  # start from an enc+dec adapter; its encoder LoRA stays frozen
        add_lora(model, ENCODER_LORA, rank=encoder_rank, alpha=2 * encoder_rank, dropout=0.0)
    if grad_ckpt:  # G x clips sequences per backward pass: activations dominate memory
        checkpoint_layers(model.llama_decoder.layers)
    state = torch.load(init, map_location=device, weights_only=True)
    unknown = set(state) - {n for n, _ in model.named_parameters()}
    if unknown:  # e.g. an enc+dec init without --encoder-rank: refuse instead of silently dropping
        msg = f"{len(unknown)} init tensors not in the model, e.g. {sorted(unknown)[:2]}"
        raise KeyError(msg)
    model.load_state_dict(state, strict=False)
    params = [
        p for n, p in model.named_parameters() if "lora_" in n and n.startswith("llama_decoder.")
    ]
    trainable = {id(p) for p in params}
    for p in model.parameters():  # new LoRA params default to trainable: freeze the encoder's
        p.requires_grad_(id(p) in trainable)
    opt = adamw(params, lr=lr, betas=(0.9, 0.98))
    model.eval()  # no dropout / masking: the policy we score is the policy we sample
    gen = NBestBeamSearch(model, group)
    mix = {k: int(v) for k, v in (x.split(":") for x in rollouts.split(",") if x)}
    if mix and sum(mix.values()) != group:
        msg = f"--rollouts sizes must sum to --group={group}"
        raise ValueError(msg)
    gens = {
        "beam": NBestBeamSearch(model, mix.get("beam", 1)),
        "sample": NBestBeamSearch(model, mix.get("sample", 1), temperature=1.0),
        "greedy": NBestBeamSearch(model, 1),
    }
    view_rng = np.random.default_rng(seed + 7)

    def mixed_groups(np_wavs: list[np.ndarray], src) -> list[list[list[int]]]:
        """Group per clip from several exploration sources; all are scored on the clean audio."""
        out: list[list[list[int]]] = [[] for _ in np_wavs]
        for name in ("beam", "sample"):
            if mix.get(name):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gs, _ = gens[name].generate_nbest(src)
                for o, g in zip(out, gs, strict=True):
                    o.extend(g)
        for speed, snr in VIEWS[1 : 1 + mix.get("aug", 0)]:  # skip the identity view
            vw = [torch.from_numpy(view(w, speed, snr, view_rng)) for w in np_wavs]
            vsrc = make_batch(vw, [[0]] * len(vw), pad, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gs, _ = gens["greedy"].generate_nbest(vsrc)
            for o, g in zip(out, gs, strict=True):
                o.extend(g)
        return out

    pad = model.target_vocab_info.pad_idx
    log = (out_dir / "log.jsonl").open("a")

    def run_eval(step: int) -> None:
        res = {k: evaluate(model, dec, v, eval_beam, device) for k, v in dev.items()}
        rec = {"step": step} | {f"{k}_{m}": r[m] for k, r in res.items() for m in ("wer", "cer")}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    run_eval(0)
    t0 = time.time()
    for step in range(1, steps + 1):
        n_unl = round(clips * unlabeled_frac) if upool is not None else 0
        n_gold = clips - n_unl
        n_scr = round(n_gold * scripted_frac)
        rng_s = np.random.default_rng(seed * 100_000 + step)
        parts = []
        if n_scr and sample_w is None:  # same draws as before (arms stay comparable)
            parts.append(pools["scripted"].sample(n=n_scr, seed=seed * 100_000 + step))
        elif n_scr:  # curriculum on difficulty (GLM-4.5 / Kimi k1.5 style) + uniform floor
            scr = pools["scripted"]
            p = (1 - difficulty_floor) * sample_w + difficulty_floor / scr.height
            parts.append(scr[rng_s.choice(scr.height, size=n_scr, replace=False, p=p)])
        if n_gold - n_scr:
            parts.append(pools["spont_gold"].sample(n=n_gold - n_scr, seed=seed * 100_000 + step))
        picks = pl.concat(parts)
        refs = picks["target"].to_list()
        if n_unl:
            rng = np.random.default_rng(seed * 100_000 + step)
            probs = upool["w"].to_numpy() / upool["w"].sum()
            uidx = rng.choice(upool.height, size=n_unl, replace=False, p=probs)
            upicks = upool[uidx]
            picks = pl.concat([picks.select("wav"), upicks.select("wav")])
            ulps = load_posteriors(
                ARTIFACTS / "posteriors" / unlabeled, upicks["clip_id"].to_list()
            )
        n_ref = len(refs)
        np_wavs = [load(w) for w in picks["wav"].to_list()]
        wavs = [torch.from_numpy(w) for w in np_wavs]
        if objective == "sft":  # control: cross-entropy on the same clips, no rollouts
            gold = make_batch(wavs, [enc(t).tolist() for t in refs], pad, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce = model(gold) / n_ref
            opt.zero_grad(set_to_none=True)
            ce.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            rec = {
                "step": step,
                "ce": float(ce),
                "gnorm": float(gnorm),
                "min": (time.time() - t0) / 60,
            }
            log.write(json.dumps(rec) + "\n")
            if step % 10 == 0:
                print(json.dumps(rec), flush=True)
            if step % eval_every == 0 or step == steps:
                run_eval(step)
                torch.save(
                    {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                    out_dir / f"step{step}_trainable.pt",
                )
            continue
        src = make_batch(wavs, [enc(t).tolist() for t in refs] + [[0]] * n_unl, pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            groups = mixed_groups(np_wavs, src) if mix else gen.generate_nbest(src)[0]
        hyps = [[normalize(dec(torch.tensor(h))) for h in g] for g in groups]
        rows_r = [[reward(r, h) for h in g] for r, g in zip(refs, hyps[:n_ref], strict=True)]
        if n_unl:
            rows_r += [
                [verifier_reward(lp, h, lm10) for h in g]
                for lp, g in zip(ulps, hyps[n_ref:], strict=True)
            ]
        rewards = torch.tensor(rows_r)
        adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-4)
        # GLM-5 "drop noisy samples": groups without signal (all hypotheses equally good) or whose
        # best hypothesis is still far from the reference (likely a misread prompt) get no update.
        noisy = torch.zeros(len(groups), dtype=torch.bool)
        noisy[:n_ref] = rewards[:n_ref].max(1).values <= -max_best_err  # gold rows only
        keep = (rewards.std(1) > 1e-6) & ~noisy
        adv = adv * keep.unsqueeze(1)
        flat = [toks for g in groups for toks in g]
        lens = torch.tensor([max(1, len(t)) + 1 for t in flat], device=device, dtype=torch.float)
        rep = make_batch([w for w in wavs for _ in range(group)], flat, pad, device)
        uniq = rep.source_seqs[::group]
        with (
            torch.no_grad(),
            cached_audio(model, uniq, src.source_seq_lens, group),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            old = sequence_logprobs(model, rep)
        stats = []
        for _ in range(inner if objective == "gspo" else 1):
            with (
                cached_audio(model, uniq, src.source_seq_lens, group),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                new = sequence_logprobs(model, rep)
            if objective == "gspo":
                ratio = torch.exp((new - old) / lens).view(len(groups), group)
                a = adv.to(device)
                if update == "mask":  # double-sided trust region: outliers leave the gradient
                    inside = (ratio > 1 - clip_eps) & (ratio < 1 + clip_eps)
                    pg = -(ratio * a * inside).sum() / inside.sum().clamp(min=1)
                else:
                    pg = -torch.min(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a).mean()
            elif objective == "drgrpo":
                # Dr. GRPO (Liu et al., 2025, arXiv:2503.20783): mean-centred reward, no std or
                # length normalisation, token-SUM log-prob. Beam hypotheses share prefixes; with
                # sum_i a_i = 0 the gradient on shared tokens cancels exactly, so credit lands on
                # the tokens where hypotheses diverge (LITERATURE §g.12).
                centred = (rewards - rewards.mean(1, keepdim=True)) * keep.unsqueeze(1)
                pg = -(centred.to(device) * new.view(len(groups), group)).sum(1).mean()
            else:  # MWER: expected centred error under the renormalised N-best posterior
                post = torch.softmax(new.view(len(groups), group), dim=1)
                err = -rewards.to(device)
                centred = (err - err.mean(1, keepdim=True)) * keep.to(device).unsqueeze(1)
                pg = (post * centred).sum(1).mean()
            gold = make_batch(wavs[:n_ref], [enc(t).tolist() for t in refs], pad, device)
            with (
                cached_audio(model, gold.source_seqs, gold.source_seq_lens, 1),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                ce = model(gold) / n_ref  # anchor on gold references only
            loss = pg + ce_weight * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            stats.append((float(pg), float(ce), float(gnorm)))
        rec = {
            "step": step,
            "reward_gold": float(rewards[:n_ref].mean()),
            "reward_unl": float(rewards[n_ref:].mean()) if n_unl else None,
            # over-generation watch: mean hypothesis/reference length on gold clips
            "len_ratio": float(
                np.mean(
                    [len(h) / max(1, len(r)) for r, g in zip(refs, hyps, strict=False) for h in g]
                )
            ),
            "best_of_group": float(rewards.max(1).values.mean()),
            "groups_kept": float(keep.float().mean()),
            "pg": float(np.mean([s[0] for s in stats])),
            "ce": float(np.mean([s[1] for s in stats])),
            "gnorm": float(np.mean([s[2] for s in stats])),
            "min": (time.time() - t0) / 60,
        }
        log.write(json.dumps(rec) + "\n")
        if step % 10 == 0:
            print(json.dumps(rec), flush=True)
        if step % eval_every == 0 or step == steps:
            run_eval(step)
            torch.save(
                {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                out_dir / f"step{step}_trainable.pt",
            )


if __name__ == "__main__":
    typer.run(main)
