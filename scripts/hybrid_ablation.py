"""Ablation and tuning-noise study of the hybrid rescorer (``joint_rescore2.py``).

Features are built once per domain on the perturbed-view pool; every variant then re-runs the
weight search from scratch with several seeds (same protocol as the hybrid: scripted weights tuned
on dev unseen-sentence clips, spontaneous weights on dev_spont; OOD never tuned on):

* feature ablations: one feature (or group) is removed from the score (its column zeroed);
* candidate ablations: candidates from a source are removed from the pool (clips left without
  candidates fall back to the full pool, which only affects the > 39 s CTC-only clips);
* the seeds of the full model measure how much the random weight search alone moves WER.

    uv run python scripts/hybrid_ablation.py --cands pool_dev_aug --post ctc1b_v3_soup_wise0.85 \
        --out reports/hybrid_ablation.csv
"""

from __future__ import annotations

import os
import re
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from joint_rescore2 import DOMAINS, build_features, pick, tune
from qxp_asr.metrics import official_normalize, wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS

SETS = {
    "dev_unseen": ("scripted", (pl.col("role") == "dev_scripted") & pl.col("unseen_sentence")),
    "itest_unseen": ("scripted", (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence")),
    "dev_spont": ("spont", pl.col("role") == "dev_spont"),
    "itest_spont": ("spont", pl.col("role") == "itest_spont"),
    "ood": ("spont", pl.col("role") == "ood"),
}
# feature columns of build_features(view_feature=True): ctc, chr, len, lp_v1, lp_v2, lp_zs, naug
FEATURE_ABLATIONS = {
    "full": [],
    "- CTC log-likelihood": ["ctc"],
    "- char 10-gram LM": ["chr"],
    "- length": ["len"],
    "- all LLM likelihoods": ["lp_v1", "lp_v2", "lp_zs"],
    "- zero-shot likelihood": ["lp_zs"],
    "- LoRA v1 likelihood": ["lp_v1"],
    "- LoRA v2 likelihood": ["lp_v2"],
    "- view-count feature": ["naug"],
}
# a candidate is kept if at least one of its proposing sources is kept
CANDIDATE_ABLATIONS = {
    "- perturbed-view candidates": lambda s: not s.endswith(("@aug", "@smp")),
    "- CTC candidates": lambda s: s != "ctc",
    "- LLM beam candidates": lambda s: s == "ctc" or s.endswith(("@aug", "@smp")),
    "CTC candidates only": lambda s: s == "ctc",
}
_F: dict = {}


def _variant(feats: dict, cols: list[str], drop: list[str], keep_src, sources: dict) -> dict:
    idx = [cols.index(c) for c in drop]
    out = {}
    for cid, (all_texts, f, risk) in feats.items():
        texts, g = all_texts, f.copy()
        g[:, idx] = 0.0
        if keep_src is not None:
            m = np.array([any(keep_src(s) for s in sources[(cid, t)]) for t in all_texts])
            if m.any():
                texts, g = [t for t, k in zip(all_texts, m, strict=True) if k], g[m]
        out[cid] = (texts, g, risk)
    return out


def _run(job: tuple) -> tuple[list[dict], list[dict]]:
    name, seed = job
    rows, hyps = [], []
    for domain in DOMAINS:
        feats, dev, evals = _F[(name, domain)], _F[("dev", domain)], _F[("eval", domain)]
        w = tune(feats, dev, feats_width(feats) - 1, _F["trials"], seed)
        for set_name, d in evals.items():
            hyp = pick(feats, d["clip_id"].to_list(), w)
            refs = d["text"].to_list()
            hyps += [
                {"variant": name, "seed": seed, "set": set_name, "clip_id": c, "hyp": h}
                for c, h in zip(d["clip_id"].to_list(), hyp, strict=True)
            ]
            row = {"variant": name, "seed": seed, "set": set_name}
            row |= {k: v for k, v in wer_cer(refs, hyp).items() if k in ("wer", "cer")}
            if seed == 0:  # oracle = best candidate in the (ablated) pool
                best = [min_err_text(feats, c, r) for c, r in zip(d["clip_id"], refs, strict=True)]
                row["oracle_wer"] = wer_cer(refs, best)["wer"]
            rows.append(row)
    return rows, hyps


def feats_width(feats: dict) -> int:
    return next(iter(feats.values()))[1].shape[1]


def min_err_text(feats: dict, cid: str, ref: str) -> str:
    if cid not in feats:
        return ""
    texts, r = feats[cid][0], official_normalize(ref).split()
    errs = [Levenshtein.distance(r, official_normalize(t).split()) for t in texts]
    return texts[int(np.argmin(errs))]


def main(
    cands: str = typer.Option("pool_dev_aug"),
    post: str = typer.Option("ctc1b_v3_soup_wise0.85"),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("indomain"),
    seeds: int = typer.Option(3),
    trials: int = typer.Option(400),
    workers: int = typer.Option(8),
    out: Path = typer.Option(Path("reports/hybrid_ablation.csv")),
    only: str = typer.Option("", help="comma-separated variant names to run (default: all)"),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    cols = ["ctc", "chr", "len", *lp_cols, "naug"]
    sources = {  # "v2@aug3" -> "v2@aug"
        (c, t): [re.sub(r"\d+$", "", s) for s in src]
        for c, t, src in table.select("clip_id", "text", "sources").iter_rows()
    }
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("text").is_not_null())
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    variants = {k: (v, None) for k, v in FEATURE_ABLATIONS.items()}
    variants |= {k: ([], v) for k, v in CANDIDATE_ABLATIONS.items()}
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",")}
    _F["trials"] = trials
    for domain, (dev_role, _) in DOMAINS.items():
        evals = {k: man.filter(e) for k, (dom, e) in SETS.items() if dom == domain}
        ids = pl.concat([d.select("clip_id") for d in evals.values()])["clip_id"].to_list()
        feats = build_features(table, ids, post, char_lm, workers, view_feature=True)
        dev = man.filter(pl.col("role") == dev_role)
        _F[("dev", domain)] = dev.filter(pl.col("unseen_sentence")) if domain == "scripted" else dev
        _F[("eval", domain)] = evals
        for name, (drop, keep_src) in variants.items():
            _F[(name, domain)] = _variant(feats, cols, drop, keep_src, sources)
        print(f"[{domain}] features for {len(feats)} clips", flush=True)
    jobs = [(name, s) for name in variants for s in range(seeds)]
    with get_context("fork").Pool(min(workers, os.cpu_count() or 1)) as pool:
        results = pool.map(_run, jobs)
    rows = [r for rs, _ in results for r in rs]
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(out)
    # per-clip outputs of every variant and seed, for paired tests (scripts/paper_stats.py)
    hyps = pl.DataFrame([h for _, hs in results for h in hs])
    hyps.write_parquet(ARTIFACTS / "preds" / f"ablation_{out.stem}.parquet")
    summary = (
        df.group_by("variant", "set", maintain_order=True)
        .agg(wer=pl.col("wer").mean(), sd=pl.col("wer").std(), oracle=pl.col("oracle_wer").mean())
        .with_columns(pl.col("wer", "sd", "oracle").round(2))
    )
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(summary.pivot("set", index="variant", values="wer"))
        print(summary.pivot("set", index="variant", values="sd"))
        print(summary.pivot("set", index="variant", values="oracle"))


if __name__ == "__main__":
    typer.run(main)
