"""Word-level verifier, step 1: confusion-network features for every competing word.

DeepSeekMath-V2-style verification for ASR (LITERATURE §g.12): instead of scoring whole
transcripts (an outcome reward, which did not beat the existing ~70 % ranking ceiling), judge each
WORD — a process-reward-style, dense signal with many more labels per clip. Per clip, all pooled
candidates (CTC + every LLM adapter) are aligned into a confusion network around the hybrid 1-best
(``qxp_asr.combine.confusion_network``); each competing word of each slot gets features that no
single recogniser has:

* hybrid-posterior vote at three temperatures, plain vote share, rank of its best voter;
* which systems' own 1-best chose it (zs, v1, v2, CTC ...), whether the hybrid 1-best chose it;
* lexical plausibility: in the training vocabulary, character-LM log-prob per character, length;
* slot context: number of alternatives, posterior entropy, the pivot's posterior.

Label (labelled clips): the reference, aligned into the same network, chose this word. Clips are
written with their role so ``scripts/word_verifier.py`` can cross-validate across speaker sets.

Run inside ``envs/omni`` (CPU):
    uv run python scripts/word_features.py --cands pool_dev_in --post ctc1b_v3_soup_wise0.85 \
        --weights hybrid_dev_in.weights.json --out wordfeat_dev_in
"""

from __future__ import annotations

import json
import math
from multiprocessing import get_context

import kenlm
import numpy as np
import polars as pl
import typer

from joint_rescore2 import build_features
from qxp_asr.combine import EPS, confusion_network
from qxp_asr.decode import ctc_loglik, load_posteriors
from qxp_asr.metrics import official_normalize
from qxp_asr.paths import ARTIFACTS, MANIFESTS

ROLES = "dev_scripted,dev_spont,itest_scripted,itest_spont,ood,test"
TEMPS = (1.0, 5.0, 20.0)


def _acoustic(job: tuple) -> list[float]:
    """CTC log-likelihood change of the whole transcript when slot k holds word w (0 for pivot)."""
    post_dir, cid, pivot, edits = job
    lp = load_posteriors(post_dir, [cid])[0]
    base = ctc_loglik(lp, " ".join(w for w in pivot if w))
    out = []
    for k, w in edits:
        words = list(pivot)
        words[k] = w
        out.append(ctc_loglik(lp, " ".join(x for x in words if x)) - base)
    return out


def softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = (x - x.max()) / t
    e = np.exp(z)
    return e / e.sum()


def main(
    cands: str = typer.Option(...),
    post: str = typer.Option(...),
    weights: str = typer.Option(..., help="tuned hybrid weights json (defines the pivot)"),
    out: str = typer.Option(...),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("indomain"),
    roles: str = typer.Option(ROLES),
    workers: int = typer.Option(8),
    acoustic: bool = typer.Option(
        False, help="add ctc_delta: CTC log-lik change of the transcript with this word in its slot"
    ),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    systems = [c.removeprefix("lp_") for c in lp_cols]
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("role").is_in(roles.split(","))
        & (pl.col("duration_s") <= 39.0)
        & pl.col("clip_id").is_in(table["clip_id"].unique())  # e.g. only the pool's clips
    )
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    vocab = set((ARTIFACTS / "lm" / lm / f"{lm_text}.unigrams.txt").read_text().split())
    lm10 = kenlm.Model(char_lm)
    saved = json.loads((ARTIFACTS / "cands" / weights).read_text())
    feats = build_features(table, man["clip_id"].to_list(), post, char_lm, workers)
    srcs = {(c, t): s for c, t, s in table.select("clip_id", "text", "sources").iter_rows()}
    lm_cache: dict[str, float] = {}

    def lm_per_char(word: str) -> float:
        if word not in lm_cache:
            score = lm10.score(" ".join(word), bos=False, eos=False) * math.log(10)
            lm_cache[word] = score / max(1, len(word))
        return lm_cache[word]

    rows, clips, jobs = [], [], []
    for cid, role, text, test_type in man.select(
        "clip_id", "role", "text", "test_type"
    ).iter_rows():
        if cid not in feats:
            continue
        texts, f, _ = feats[cid]
        spont = role in ("dev_spont", "itest_spont", "ood") or test_type == "spontaneous"
        w = saved["spont" if spont else "scripted"]
        lin = np.array([w["chr"], w["len"], *[w[c] for c in lp_cols]])
        score = f[:, 0] + f[:, 1 : 3 + len(lp_cols)] @ lin
        order = np.argsort(-score)
        hyps = [texts[i] for i in order]
        s_sorted = score[order]
        posts = {t: softmax(s_sorted, t) for t in TEMPS}
        # each system's own 1-best among the candidates it proposed (CTC: best CTC log-lik)
        best_of = {}
        for j, sysname in enumerate(systems):
            own = [i for i in order if sysname in srcs.get((cid, texts[i]), [])]
            if own:
                best_of[sysname] = max(own, key=lambda i, j=j: f[i, 3 + j])
        ctc_own = [i for i in order if "ctc" in srcs.get((cid, texts[i]), [])]
        if ctc_own:
            best_of["ctc"] = max(ctc_own, key=lambda i: f[i, 0])
        rank_of = {int(i): r for r, i in enumerate(order)}
        ref = official_normalize(text) if text else None
        cn = confusion_network([*hyps, ref] if ref else hyps)
        n, ref_idx = len(hyps), len(hyps)
        pivot_words = []
        for slot in cn:
            pw = next((wd for wd, v in slot.items() if 0 in v), EPS)
            pivot_words.append("" if pw == EPS else pw)
        edits, edit_rows = [], []
        clips.append({"clip_id": cid, "role": role, "ref": ref, "pivot": hyps[0]})
        for k, slot in enumerate(cn):
            alts = {wd: {i for i in v if i < n} for wd, v in slot.items()}
            alts = {wd: v for wd, v in alts.items() if v}
            if not alts:
                continue
            p5 = np.array([posts[5.0][list(v)].sum() for v in alts.values()])
            entropy = float(-(p5 * np.log(p5 + 1e-12)).sum())
            pivot_word = next(wd for wd, v in alts.items() if 0 in v)
            for wd, voters in alts.items():
                idx = list(voters)
                rec = {
                    "clip_id": cid,
                    "role": role,
                    "slot": k,
                    "word": wd,
                    "label": None if ref is None else int(ref_idx in slot.get(wd, set())),
                    "is_eps": int(wd == EPS),
                    "in_pivot": int(wd == pivot_word),
                    "vote": len(idx) / n,
                    "best_rank": min(idx),
                    "n_cands": n,
                    "n_alts": len(alts),
                    "slot_entropy": entropy,
                    "slot_max_post": float(p5.max()),
                    "pivot_post": float(posts[5.0][0]),
                    "spont": int(spont),
                    "in_vocab": int(wd in vocab) if wd != EPS else 1,
                    "nchar": len(wd),
                    "chr_lm_pc": lm_per_char(wd) if wd != EPS else 0.0,
                }
                for t in TEMPS:
                    rec[f"post_T{t:g}"] = float(posts[t][idx].sum())
                for sysname, i in best_of.items():
                    rec[f"src_{sysname}"] = int(rank_of[int(i)] in voters)
                edits.append((k, "" if wd == EPS else wd))
                edit_rows.append(len(rows))
                rows.append(rec)
        jobs.append(((ARTIFACTS / "posteriors" / post, cid, pivot_words, edits), edit_rows))
    if acoustic:  # acoustic evidence for each competing word, in its context (exact CTC)
        with get_context("fork").Pool(workers) as pool:
            deltas = pool.map(_acoustic, [j for j, _ in jobs], chunksize=4)
        for (_, idxs), ds in zip(jobs, deltas, strict=True):
            for i, d in zip(idxs, ds, strict=True):
                rows[i]["ctc_delta"] = max(d, -500.0)
                rows[i]["ctc_delta_pc"] = max(d, -500.0) / max(1, rows[i]["nchar"])
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_parquet(ARTIFACTS / "cands" / f"{out}.parquet")
    pl.DataFrame(clips).write_parquet(ARTIFACTS / "cands" / f"{out}_clips.parquet")
    lab = df.filter(pl.col("label").is_not_null())
    print(f"{df.height} word alternatives in {len(clips)} clips; labelled {lab.height}")
    print(lab.group_by("in_pivot").agg(pl.len(), acc=pl.col("label").mean()))


if __name__ == "__main__":
    typer.run(main)
