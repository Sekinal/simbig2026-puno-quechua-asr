"""Exploration sweep for RL: which rollout source ever produces a better transcript (pass@k)?

RL can only amplify transcripts the policy already produces (VibeThinker's spectrum-to-signal
principle, arXiv:2511.06221; LITERATURE §g.12). With beam search, LoRA v2's beam-8 beats its own
top-1 on only 2.7 % of held-out-sentence clips (reports/RESULTS.md). This sweep compares rollout
sources on small fixed subsets (fast: ~10-30 min per GPU):

* ``beam``   beam-``k`` list;
* ``T<t>``   ``k`` ancestral samples at temperature t (what LLM RL uses);
* ``fork``   greedy + ``k-1`` forks at the most uncertain tokens (runner-up token forced there,
  then greedy; ``qxp_asr.nbest.generate_forks``);
* ``aug``    greedy decodes of ``k`` perturbed audio views (speed 0.85-1.15, additive noise):
  exploration in the *input* (domain randomisation) instead of the output;
* ``union``  all of the above.

Per set and source: corpus WER of the beam top-1, oracle WER over the source's hypotheses, share
of clips where some hypothesis beats top-1 (clips with policy-gradient signal), distinct count.

Run inside ``envs/omni``:
    uv run python scripts/pass_at_k.py --init .../llm7b_lora_v2/best_trainable.pt
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model

from qxp_asr.audio import load
from qxp_asr.augment import VIEWS, view
from qxp_asr.combine import medoid, rover
from qxp_asr.lora import add_lora
from qxp_asr.metrics import wer_cer
from qxp_asr.nbest import NBestBeamSearch, generate_forks, make_batch
from qxp_asr.paths import MANIFESTS
from qxp_asr.text import normalize
from train_rl import DECODER_LORA, half_of, reward


def decode(gen, model, dec, wavs: list[np.ndarray], batch: int, forks: int = 0) -> list[list[str]]:
    out = []
    for i in range(0, len(wavs), batch):
        part = [torch.from_numpy(w) for w in wavs[i : i + batch]]
        src = make_batch(part, [[0]] * len(part), model.target_vocab_info.pad_idx, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if forks:
                groups = generate_forks(model, src, forks)
            else:
                groups, _ = gen.generate_nbest(src)
        out += [[normalize(dec(torch.tensor(h))) for h in g] for g in groups]
    return out


def main(
    init: str = typer.Option(..., help="decoder-LoRA trainable.pt, or 'zs' for zero-shot"),
    card: str = typer.Option("omniASR_LLM_7B_v2"),
    k: int = typer.Option(8),
    sources: str = typer.Option("beam,T0.7,T1,T1.3,aug"),
    sets: str = typer.Option("heldout_sent_h0,dev_unseen,dev_spont,ood"),
    clips: int = typer.Option(150, help="clips per set"),
    batch: int = typer.Option(4),
    seed: int = typer.Option(0),
    out: str = typer.Option("", help="per-clip 'beats top-1' table (parquet) for paired tests"),
) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    man = pl.read_parquet(MANIFESTS / "all.parquet").with_columns(
        half=pl.col("sentence_id").map_elements(
            lambda x: half_of(x) if x else -1, return_dtype=pl.Int64
        )
    )
    conds = {
        "heldout_sent_h0": (pl.col("role") == "holdout_sent") & (pl.col("half") == 0),
        "dev_unseen": (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"),
        "dev_spont": pl.col("role") == "dev_spont",
        "ood": pl.col("role") == "ood",
    }
    tok = load_tokenizer(card)
    dec = tok.create_decoder(skip_special_tokens=True)
    model = load_model(card, device=torch.device("cuda"), dtype=torch.bfloat16)
    add_lora(model, DECODER_LORA, rank=16, alpha=32, dropout=0.0)
    if init != "zs":
        state = torch.load(init, map_location="cuda", weights_only=True)
        model.load_state_dict(state, strict=False)
    model.eval()
    beam, greedy = NBestBeamSearch(model, k), NBestBeamSearch(model, 1)
    results, per_clip = [], []
    for name in sets.split(","):
        rows = man.filter(conds[name] & pl.col("text").is_not_null() & (pl.col("duration_s") <= 20))
        rows = rows.sample(n=min(clips, rows.height), seed=seed)
        refs = [normalize(t) for t in rows["text"].to_list()]
        wavs = [load(w) for w in rows["wav"].to_list()]
        hyps: dict[str, list[list[str]]] = {"beam": decode(beam, model, dec, wavs, batch)}
        for src in sources.split(","):
            if src.startswith("T"):
                gen = NBestBeamSearch(model, k, temperature=float(src[1:]))
                hyps[src] = decode(gen, model, dec, wavs, batch)
            elif src == "fork":
                hyps["fork"] = decode(None, model, dec, wavs, batch, forks=k - 1)
            elif src == "aug":
                views = []
                for speed, snr in VIEWS[:k]:
                    vw = [view(w, speed, snr, rng) for w in wavs]
                    views.append([h[0] for h in decode(greedy, model, dec, vw, batch)])
                hyps["aug"] = [list(v) for v in zip(*views, strict=True)]
        hyps["union"] = [[x for s in list(hyps) for x in hyps[s][i]] for i in range(len(refs))]
        top1 = [h[0] for h in hyps["beam"]]
        t1_err = [-reward(r, t) for r, t in zip(refs, top1, strict=True)]
        for src, hs in hyps.items():
            best, beats, distinct = [], 0, 0
            mbr = [medoid(h) for h in hs]  # self-consistency: the most central hypothesis
            vote = [rover([h[0], *h]) for h in hs]  # word-level vote, top-1 as pivot
            for ref, h, e1 in zip(refs, hs, t1_err, strict=True):
                errs = [-reward(ref, x) for x in h]
                best.append(h[int(np.argmin(errs))])
                beats += min(errs) < e1 - 1e-9
                distinct += len(set(h))
            per_clip += [
                {
                    "set": name,
                    "clip_id": c,
                    "source": src,
                    "beats": min(-reward(ref, x) for x in h) < e1 - 1e-9,
                }
                for c, ref, h, e1 in zip(rows["clip_id"], refs, hs, t1_err, strict=True)
            ]
            results.append(
                {
                    "set": name,
                    "source": src,
                    "top1_wer": round(wer_cer(refs, top1)["wer"], 2),
                    "oracle_wer": round(wer_cer(refs, best)["wer"], 2),
                    "mbr_wer": round(wer_cer(refs, mbr)["wer"], 2),
                    "rover_wer": round(wer_cer(refs, vote)["wer"], 2),
                    "beats_top1_%": round(100 * beats / len(refs), 1),
                    "distinct": round(distinct / len(refs), 1),
                }
            )
        print(pl.DataFrame(results).filter(pl.col("set") == name), flush=True)
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(pl.DataFrame(results))
    if out:
        pl.DataFrame(per_clip).write_parquet(out)


if __name__ == "__main__":
    typer.run(main)
