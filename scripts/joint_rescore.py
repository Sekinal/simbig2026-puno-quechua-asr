"""Joint CTC + LLM-ASR hypothesis rescoring (hybrid CTC/attention-style system combination).

Inspiration: joint CTC/attention decoding and rescoring (Watanabe et al., 2017, "Hybrid
CTC/Attention Architecture for End-to-End Speech Recognition", IEEE JSTSP), where a CTC model's
alignment-free likelihood re-ranks hypotheses of an autoregressive decoder; N-best rescoring with
an external LM (docs/LITERATURE.md §g.2).

For every clip, candidates = CTC word-LM beam N-best + the hypotheses of every ``--llm`` system
(LoRA-adapted and zero-shot LLM-7B). Each candidate gets
    score = log P_ctc(text | audio) + gamma * ln P_char10(text) + delta * n_chars + mu * [LLM hyp]
where log P_ctc is the exact CTC log-likelihood (all alignments) from our CTC posteriors. Weights
are tuned per domain on dev (scripted: unseen-sentence clips) and applied unchanged to itest, OOD
and test.

Run inside ``envs/omni`` (CPU):
    uv run python scripts/joint_rescore.py --post ctc1b_v2 --llm llm7b_lora_v1 --llm zs_llm7b_all
"""

from __future__ import annotations

import itertools
import math
from multiprocessing import get_context

import kenlm
import numpy as np
import polars as pl
import typer
from pyctcdecode import build_ctcdecoder

from qxp_asr.decode import ctc_loglik, load_posteriors
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.text import normalize
from qxp_asr.vocab import pyctcdecode_labels

LN10 = math.log(10)
DOMAINS = {"scripted": ("dev_scripted", "itest_scripted"), "spont": ("dev_spont", "itest_spont")}
TEST_TYPE = {"scripted": "scripted", "spont": "spontaneous"}


def features(args: tuple) -> list[tuple[str, float, float, int, int]]:
    """(text, ctc, charlm, n_chars, is_llm) per unique candidate of one clip."""
    lp, cands, char_lm_path = args
    lm = _LM.setdefault(char_lm_path, kenlm.Model(char_lm_path))
    out = []
    for text, is_llm in cands:
        chars = " ".join("|" if c == " " else c for c in text)
        out.append((text, ctc_loglik(lp, text), lm.score(chars) * LN10, len(text), is_llm))
    return out


_LM: dict[str, kenlm.Model] = {}


def pick(feats: list, gamma: float, delta: float, mu: float) -> list[str]:
    best = []
    for clip in feats:
        scores = [c + gamma * ch + delta * n + mu * f for _, c, ch, n, f in clip]
        best.append(clip[int(np.argmax(scores))][0])
    return best


def floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",")]


def main(
    post: str = typer.Option("ctc1b_v2"),
    llm: list[str] = typer.Option(
        ["llm7b_lora_v1", "zs_llm7b_all"], help="LLM-ASR prediction sets added as candidates"
    ),
    lm: str = typer.Option("dev_v2"),
    alpha: float = typer.Option(0.5, help="word-LM weight for generating the CTC N-best"),
    keep: int = typer.Option(16),
    gammas: str = typer.Option("0,0.2,0.4,0.7,1.0"),
    deltas: str = typer.Option("0,0.5,1,2"),
    mus: str = typer.Option("0,2,5,10,20,50"),
    workers: int = typer.Option(4),
    name: str = typer.Option("joint_v1"),
    fixed_scripted: str = typer.Option(
        "", help="gamma,delta,mu for scripted (final mode: dev sentences are no longer unseen)"
    ),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    lm_dir = ARTIFACTS / "lm" / lm
    decoder = build_ctcdecoder(
        pyctcdecode_labels(),
        str(lm_dir / "all.word4.bin"),
        unigrams=(lm_dir / "all.unigrams.txt").read_text().split(),
        alpha=alpha,
        beta=0.0,
    )
    hyps = {}
    for n in llm:
        tab = pl.read_parquet(ARTIFACTS / "preds" / f"{n}.parquet")
        hyps[n] = dict(zip(tab["clip_id"], (normalize(h or "") for h in tab["hyp"]), strict=True))
    char_lm = str(lm_dir / "all.char10.bin")

    def build(rows: pl.DataFrame) -> list:
        ids = rows["clip_id"].to_list()
        lps = load_posteriors(ARTIFACTS / "posteriors" / post, ids)
        with get_context("fork").Pool(workers) as pool:
            beams = decoder.decode_beams_batch(pool, lps, beam_width=64)
            jobs = []
            for cid, lp, bs in zip(ids, lps, beams, strict=True):
                cands = {" ".join(b[0].split()): 0 for b in bs[:keep]}
                for n in llm:
                    if hyps[n].get(cid):
                        cands[hyps[n][cid]] = 1
                jobs.append((lp, [(t, f) for t, f in cands.items() if t], char_lm))
            return pool.map(features, jobs, chunksize=8)

    outputs, results = [], []
    for domain, (dev_role, itest_role) in DOMAINS.items():
        if domain == "scripted" and fixed_scripted:
            g, d, m = floats(fixed_scripted)
            print(f"[{domain}] fixed gamma={g} delta={d} mu={m}", flush=True)
        else:
            dev = man.filter(pl.col("role") == dev_role)
            if domain == "scripted":
                dev = dev.filter(pl.col("unseen_sentence"))
            feats = build(dev)
            refs = dev["text"].to_list()
            best = (math.inf, None)
            for g, d, m in itertools.product(floats(gammas), floats(deltas), floats(mus)):
                s = wer_cer(refs, pick(feats, g, d, m))
                if s["wer"] + 0.5 * s["cer"] < best[0]:
                    best = (s["wer"] + 0.5 * s["cer"], (g, d, m), s)
            g, d, m = best[1]
            print(f"[{domain}] gamma={g} delta={d} mu={m} dev {best[2]}", flush=True)
        extra = ["ood"] if domain == "spont" else []
        apply = man.filter(
            pl.col("role").is_in([dev_role, itest_role, *extra])
            | ((pl.col("role") == "test") & (pl.col("test_type") == TEST_TYPE[domain]))
        )
        out = apply.select("clip_id", "role", "unseen_sentence", "text", "test_type").with_columns(
            hyp=pl.Series(pick(build(apply), g, d, m))
        )
        outputs.append(out)
        lab = out.filter(pl.col("text").is_not_null())
        results += [
            {"domain": domain, "role": r, "unseen": u}
            | wer_cer(x["text"].to_list(), x["hyp"].to_list())
            for (r, u), x in lab.group_by("role", "unseen_sentence")
        ]
    pl.concat(outputs).write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
    with pl.Config(tbl_rows=30):
        print(pl.DataFrame(results).sort("role", "unseen"))


if __name__ == "__main__":
    typer.run(main)
