"""N-best rescoring of CTC beam search with a high-order character LM (open vocabulary).

Pipeline per domain: pyctcdecode word-LM beam search (KenLM 4-gram, weight ``alpha``) produces an
N-best list; every hypothesis is rescored with
    combined_score + gamma * ln(10) * log10 P_char10(text) + delta * n_chars
and the best one is kept. The character 10-gram scores agglutinated word forms never seen as
whole words (spontaneous OOV is 16-24 %), which a word LM cannot (docs/LITERATURE.md §g.2:
character/subword LMs for morphologically rich languages; N-best rescoring as in e.g.
Hrinchuk et al. 2020 / Salazar et al. 2020).

(alpha, gamma, delta) are tuned on ``dev_<domain>`` (scripted: unseen-sentence clips only) and
applied unchanged to itest, OOD (spontaneous settings) and the shared-task test.

Run inside ``envs/omni`` (kenlm built with MAX_ORDER=12):
    uv run python /workspace/simbig/repo/scripts/rescore.py --post ctc1b_v2 --lm dev_v2
"""

from __future__ import annotations

import itertools
import json
import math
from multiprocessing import get_context
from pathlib import Path

import kenlm
import numpy as np
import polars as pl
import typer
from pyctcdecode import build_ctcdecoder

from qxp_asr.decode import greedy, load_posteriors
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.vocab import pyctcdecode_labels

LN10 = math.log(10)
DOMAINS = {"scripted": ("dev_scripted", "itest_scripted"), "spont": ("dev_spont", "itest_spont")}
TEST_TYPE = {"scripted": "scripted", "spont": "spontaneous"}

Beams = list[list[tuple[str, float]]]  # per clip: (text, combined word-LM score)


def nbest(decoder, lps: list[np.ndarray], width: int, keep: int, workers: int) -> Beams:
    with get_context("fork").Pool(workers) as pool:
        out = decoder.decode_beams_batch(pool, lps, beam_width=width)
    return [[(" ".join(b[0].split()), float(b[-1])) for b in beams[:keep]] for beams in out]


def char_scores(lm: kenlm.Model, beams: Beams) -> list[list[float]]:
    def score(text: str) -> float:
        chars = " ".join("|" if c == " " else c for c in text)
        return lm.score(chars, bos=True, eos=True) * LN10

    return [[score(t) for t, _ in clip] for clip in beams]


def pick(beams: Beams, cs: list[list[float]], gamma: float, delta: float) -> list[str]:
    out = []
    for clip, scores in zip(beams, cs, strict=True):
        if not clip:
            out.append("")
            continue
        totals = [s + gamma * c + delta * len(t) for (t, s), c in zip(clip, scores, strict=True)]
        out.append(clip[int(np.argmax(totals))][0])
    return out


def floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",")]


def main(
    post: str = typer.Option(...),
    lm: str = typer.Option("dev_v2"),
    word_lm: str = typer.Option("all"),
    char_lm: str = typer.Option("all"),
    alphas: str = typer.Option("0.2,0.4,0.6"),
    gammas: str = typer.Option("0,0.1,0.2,0.3,0.5,0.8"),
    deltas: str = typer.Option("0,0.5,1,1.5"),
    beam_width: int = typer.Option(64),
    keep: int = typer.Option(32),
    workers: int = typer.Option(4),
    tag: str = typer.Option(""),
    tune_on: str = typer.Option(
        "unseen", help="scripted tuning subset: unseen (default) | all (org-mode comparison)"
    ),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    post_dir, lm_dir = ARTIFACTS / "posteriors" / post, ARTIFACTS / "lm" / lm
    unigrams = (lm_dir / f"{word_lm}.unigrams.txt").read_text().split()
    decoder = build_ctcdecoder(
        pyctcdecode_labels(), str(lm_dir / f"{word_lm}.word4.bin"), unigrams=unigrams
    )
    clm = kenlm.Model(str(lm_dir / f"{char_lm}.char10.bin"))
    results, outputs, chosen = [], [], {}
    for domain, (dev_role, itest_role) in DOMAINS.items():
        dev = man.filter(pl.col("role") == dev_role)
        if domain == "scripted" and tune_on == "unseen":
            dev = dev.filter(pl.col("unseen_sentence"))
        refs = dev["text"].to_list()
        dev_lp = load_posteriors(post_dir, dev["clip_id"].to_list())
        print(f"[{domain}] greedy {wer_cer(refs, greedy(dev_lp))}")
        best = (math.inf, None)
        for alpha in floats(alphas):
            decoder.reset_params(alpha=alpha, beta=0.0)
            beams = nbest(decoder, dev_lp, beam_width, keep, workers)
            cs = char_scores(clm, beams)
            for gamma, delta in itertools.product(floats(gammas), floats(deltas)):
                s = wer_cer(refs, pick(beams, cs, gamma, delta))
                if s["wer"] + 0.5 * s["cer"] < best[0]:  # favour CER too: both are scored
                    best = (s["wer"] + 0.5 * s["cer"], (alpha, gamma, delta), s)
            print(f"  alpha={alpha}: best so far {best[1]} {best[2]}", flush=True)
        alpha, gamma, delta = best[1]
        chosen[domain] = {"alpha": alpha, "gamma": gamma, "delta": delta}
        decoder.reset_params(alpha=alpha, beta=0.0)
        extra = ["ood"] if domain == "spont" else []
        apply = man.filter(
            pl.col("role").is_in([dev_role, itest_role, *extra])
            | ((pl.col("role") == "test") & (pl.col("test_type") == TEST_TYPE[domain]))
        )
        lps = load_posteriors(post_dir, apply["clip_id"].to_list())
        beams = nbest(decoder, lps, beam_width, keep, workers)
        hyps = pick(beams, char_scores(clm, beams), gamma, delta)
        out = apply.select("clip_id", "role", "unseen_sentence", "text", "test_type").with_columns(
            hyp=pl.Series(hyps), greedy=pl.Series(greedy(lps))
        )
        outputs.append(out)
        for (role, unseen), g in out.filter(pl.col("text").is_not_null()).group_by(
            "role", "unseen_sentence"
        ):
            results.extend(
                {"domain": domain, "role": role, "unseen": unseen, "decode": kind}
                | wer_cer(g["text"].to_list(), g[kind].to_list())
                for kind in ("greedy", "hyp")
            )
    name = f"{post}__rescore_{word_lm}_{char_lm}{tag}"
    pl.concat(outputs).write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
    table = pl.DataFrame(results).sort("domain", "role", "unseen", "decode")
    Path(ARTIFACTS / "preds" / f"{name}.json").write_text(
        json.dumps({"params": chosen, "results": table.to_dicts()}, indent=1)
    )
    print(chosen)
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(table)


if __name__ == "__main__":
    typer.run(main)
