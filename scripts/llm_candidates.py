"""Pool LLM-ASR N-best candidates with CTC candidates and score every candidate under each adapter.

Hybrid CTC/attention rescoring (Watanabe et al., 2017) needs the attention (here: LLM-ASR)
log-likelihood of *every* candidate, not just a flag for the LLM's own 1-best. For each adapter
(``zs`` = zero-shot, i.e. LoRA disabled; otherwise a decoder-LoRA ``*_trainable.pt``):

1. beam-search N-best (``--beam``) for every clip -> added to the candidate pool;
2. after pooling, the exact sequence log-probability (incl. EOS) of every candidate, with the 7B
   encoder computed once per clip and shared by all its candidates.

The 7B base model is loaded once; adapters are swapped by loading LoRA weights (zero-shot = all
``lora_b`` set to zero, which makes every LoRA update vanish).

Output ``artifacts/cands/<name>.parquet``: clip_id, text, sources (list), lp_<adapter> columns.

Run inside ``envs/omni``:
    uv run python scripts/llm_candidates.py --ctc ctc_v3 --name pool_v1 \
        --adapter zs --adapter v1=.../llm7b_lora_v1/best_trainable.pt --adapter v2=...
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model

from qxp_asr.audio import load
from qxp_asr.augment import VIEWS, view
from qxp_asr.lora import add_lora
from qxp_asr.nbest import NBestBeamSearch, cached_audio, make_batch, sequence_logprobs
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import SOURCES
from qxp_asr.speed import enable_fast_math
from qxp_asr.splits import crossfit_heldout, with_crossfit_folds
from qxp_asr.text import normalize

DECODER_LORA = (
    r"llama_decoder\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
)


ENCODER_LORA = (
    r"encoder\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
)


def set_adapter(model, path: str | None) -> None:
    """Load an adapter; LoRA modules it has no weights for (e.g. the encoder LoRA of a
    decoder-only adapter, or everything for zero-shot) get B = 0, i.e. no update."""
    lora = {n: p for n, p in model.named_parameters() if "lora_" in n}
    state = torch.load(path, map_location="cuda", weights_only=True) if path else {}
    with torch.no_grad():
        for n, p in lora.items():
            if n in state:
                p.copy_(state[n])
            elif "lora_b" in n:
                p.zero_()


def main(
    ctc: str = typer.Option(
        "", help="candidate table from scripts/ctc_nbest.py (or an earlier pool to extend)"
    ),
    name: str = typer.Option(...),
    adapter: list[str] = typer.Option(..., help="'zs' or name=path/to/trainable.pt"),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    beam: int = typer.Option(8),
    nbest: str = typer.Option(
        "",
        help="adapters that add beam candidates, comma-separated (default: all; others only score)",
    ),
    batch_seconds: float = typer.Option(
        60.0, help="audio seconds per N-best batch (x beam sequences); long clips go alone"
    ),
    max_s: float = typer.Option(39.0, help="longer clips keep CTC candidates only"),
    base: str = typer.Option(
        "", help="earlier pool whose LLM candidates/scores are kept; --ctc replaces its CTC ones"
    ),
    crossfit_fold: int = typer.Option(
        -1, help="instead of --ctc: gold training clips held out of fold k (qxp_asr.splits)"
    ),
    max_clips: int = typer.Option(0, help="with --crossfit-fold: subsample read-speech clips"),
    score: bool = typer.Option(True, help="--no-score: only collect N-best candidates"),
    views: int = typer.Option(
        0, help="also add greedy decodes of this many perturbed audio views (qxp_asr.augment)"
    ),
    view_adapters: str = typer.Option("zs,v2", help="adapters that decode the perturbed views"),
    view_start: int = typer.Option(0, help="first view index (extend an earlier --views pool)"),
    samples: int = typer.Option(
        0, help="control for --views: this many temperature samples of the CLEAN audio per clip"
    ),
    sample_temp: float = typer.Option(1.0, help="temperature of --samples"),
    rank: int = typer.Option(16, help="decoder LoRA rank of every adapter (64: exact soup of 4)"),
    encoder_rank: int = typer.Option(
        0, help="encoder LoRA rank (16: encoder+decoder adapters; others get no encoder update)"
    ),
    blank_audio: bool = typer.Option(
        False,
        help="score with 1 s of near-silence instead of the clip: the adapter's internal LM "
        "(text prior) for density-ratio / ILM subtraction; never reuses audio scores",
    ),
) -> None:
    enable_fast_math()
    if crossfit_fold >= 0:  # realistic errors: neither voice nor text seen by the fold model
        held = with_crossfit_folds(pl.read_parquet(MANIFESTS / "all.parquet")).filter(
            crossfit_heldout(crossfit_fold)
            & (pl.col("role") == "train")
            & (SOURCES["scripted"] | SOURCES["spont_gold"])
            & pl.col("text").is_not_null()
        )
        scr = held.filter(pl.col("corpus") == "scripted")
        if max_clips and scr.height > max_clips:
            scr = scr.sample(n=max_clips, seed=crossfit_fold)
        held = pl.concat([scr, held.filter(pl.col("corpus") == "spontaneous")])
        cands = pl.DataFrame({"clip_id": held["clip_id"], "text": [None] * held.height})
    else:
        cands = pl.read_parquet(ARTIFACTS / "cands" / f"{ctc}.parquet")
    if base:  # keep an earlier pool's LLM candidates + scores, replace its CTC candidates
        old = pl.read_parquet(ARTIFACTS / "cands" / f"{base}.parquet").with_columns(
            sources=pl.col("sources").list.set_difference(["ctc"])
        )
        new = cands.select("clip_id", "text", sources=pl.concat_list(pl.lit("ctc")))
        cands = pl.concat(
            [old.filter(pl.col("sources").list.len() > 0), new], how="diagonal_relaxed"
        )
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("clip_id").is_in(cands["clip_id"].unique().to_list())
    )
    clips = man.filter(pl.col("duration_s") <= max_s).sort("duration_s")
    pool: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if crossfit_fold >= 0:
        pass  # no seed candidates: the pool is filled by the N-best of the adapters
    elif "sources" in cands.columns:  # extending an earlier pool: keep who proposed what
        for cid, text, srcs in cands.select("clip_id", "text", "sources").iter_rows():
            pool[cid][text].update(srcs)
    else:
        for cid, text in cands.select("clip_id", "text").iter_rows():
            pool[cid][text].add("ctc")

    tok = load_tokenizer(card)
    enc, dec = tok.create_encoder(), tok.create_decoder(skip_special_tokens=True)
    model = load_model(card, device=torch.device("cuda"), dtype=torch.bfloat16)
    add_lora(model, DECODER_LORA, rank=rank, alpha=2 * rank, dropout=0.0)
    if encoder_rank:
        add_lora(model, ENCODER_LORA, rank=encoder_rank, alpha=2 * encoder_rank, dropout=0.0)
    model.eval()
    pad = model.target_vocab_info.pad_idx
    gen = NBestBeamSearch(model, beam)
    specs = [(a, None) if a == "zs" else tuple(a.split("=", 1)) for a in adapter]
    wav_cache = {
        c: torch.from_numpy(load(w)) for c, w in clips.select("clip_id", "wav").iter_rows()
    }
    ids = clips["clip_id"].to_list()
    durs = dict(zip(ids, clips["duration_s"].to_list(), strict=True))

    t0 = time.time()
    generators = set(nbest.split(",")) if nbest else {a for a, _ in specs}
    for aname, path in specs:  # 1) N-best from every (generating) adapter
        if aname not in generators:
            continue
        set_adapter(model, path)
        for part in _batches(ids, durs, batch_seconds):
            batch = make_batch([wav_cache[c] for c in part], [[0]] * len(part), pad, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                groups, _ = gen.generate_nbest(batch)
            for cid, g in zip(part, groups, strict=True):
                for toks in g:
                    text = normalize(dec(torch.tensor(toks)))
                    if text:
                        pool[cid][text].add(aname)
        print(f"[{aname}] n-best done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if views:  # exploration in the input: perturbed-view decodes (reports/RESULTS.md, sweep)
        greedy = NBestBeamSearch(model, 1)
        for aname, path in specs:
            if aname not in view_adapters.split(","):
                continue
            set_adapter(model, path)
            for j, (speed, snr) in enumerate(
                VIEWS[1 + view_start : 1 + view_start + views], start=view_start
            ):
                rng = np.random.default_rng(j)
                vdurs = {c: d / speed for c, d in durs.items()}
                for part in _batches(ids, vdurs, 4 * batch_seconds):  # greedy: larger batches
                    wavs = [
                        torch.from_numpy(view(wav_cache[c].numpy(), speed, snr, rng)) for c in part
                    ]
                    batch = make_batch(wavs, [[0]] * len(part), pad, "cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        groups, _ = greedy.generate_nbest(batch)
                    for cid, g in zip(part, groups, strict=True):
                        text = normalize(dec(torch.tensor(g[0])))
                        if text:  # one tag per view: the number of agreeing views is a feature
                            pool[cid][text].add(f"{aname}@aug{j}")
            print(f"[{aname}] {views} views done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if samples:  # matched control: as many candidates as views, but from sampling, not the input
        torch.manual_seed(0)
        sampler = NBestBeamSearch(model, samples, temperature=sample_temp)
        for aname, path in specs:
            if aname not in view_adapters.split(","):
                continue
            set_adapter(model, path)
            for part in _batches(ids, durs, batch_seconds):
                batch = make_batch([wav_cache[c] for c in part], [[0]] * len(part), pad, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    groups, _ = sampler.generate_nbest(batch)
                for cid, g in zip(part, groups, strict=True):
                    for j, toks in enumerate(g):
                        text = normalize(dec(torch.tensor(toks)))
                        if text:
                            pool[cid][text].add(f"{aname}@smp{j}")
            mins = (time.time() - t0) / 60
            print(f"[{aname}] {samples} samples done ({mins:.1f} min)", flush=True)

    scores: dict[str, dict[tuple[str, str], float]] = {a: {} for a, _ in specs}
    silence = torch.from_numpy(np.random.default_rng(0).normal(0, 1e-4, 16000).astype(np.float32))
    for a, _ in specs:  # reuse scores of an extended pool; only new candidates get scored
        if f"lp_{a}" in cands.columns and not blank_audio:
            known = cands.select("clip_id", "text", f"lp_{a}").drop_nulls().iter_rows()
            scores[a] = {(c, t): v for c, t, v in known}
    for aname, path in specs if score else []:  # 2) score the pooled candidates
        set_adapter(model, path)
        for cid in ids:
            texts = sorted(t for t in pool[cid] if (cid, t) not in scores[aname])
            wav = silence if blank_audio else wav_cache[cid]
            for j in range(0, len(texts), 32):
                chunk = texts[j : j + 32]
                batch = make_batch(
                    [wav] * len(chunk), [enc(t).tolist() for t in chunk], pad, "cuda"
                )
                with (
                    torch.no_grad(),
                    cached_audio(
                        model, batch.source_seqs[:1], batch.source_seq_lens[:1], len(chunk)
                    ),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    lp = sequence_logprobs(model, batch).tolist()
                for t, v in zip(chunk, lp, strict=True):
                    scores[aname][(cid, t)] = v
        print(f"[{aname}] scoring done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    rows = [
        {"clip_id": cid, "text": t, "sources": sorted(src)}
        | {f"lp_{a}": scores[a].get((cid, t)) for a, _ in specs}
        for cid, texts in pool.items()
        for t, src in texts.items()
    ]
    out = ARTIFACTS / "cands" / f"{name}.parquet"
    pl.DataFrame(rows).write_parquet(out)
    print(f"{len(rows)} scored candidates -> {out}")


def _batches(ids: list[str], durs: dict[str, float], max_seconds: float) -> list[list[str]]:
    """Greedy duration-bounded batches (padded length x size <= max_seconds)."""
    out, cur, longest = [], [], 0.0
    for c in ids:
        longest = max(longest, durs[c]) if cur else durs[c]
        if cur and longest * (len(cur) + 1) > max_seconds:
            out.append(cur)
            cur, longest = [], durs[c]
        cur.append(c)
    if cur:
        out.append(cur)
    return out


if __name__ == "__main__":
    typer.run(main)
