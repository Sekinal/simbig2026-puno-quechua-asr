"""Hybrid CTC + LLM-ASR rescoring with LLM log-likelihoods as features (joint_rescore v2).

Every pooled candidate (CTC word-LM N-best + beam-8 N-best of each LLM adapter; built by
``scripts/ctc_nbest.py`` and ``scripts/llm_candidates.py``) gets the features

    ctc  exact CTC log-likelihood under our CTC model (all alignments)
    chr  char-10gram LM log-probability            len  number of characters
    lp_* sequence log-probability under each LLM adapter (zero-shot, LoRA v1, LoRA v2, ...)

and the score  ctc + w . features. This is hybrid CTC/attention rescoring (Watanabe et al., 2017)
with several attention scorers plus an external LM; it replaces the earlier 0/1 "LLM hypothesis"
flag by the LLMs' actual likelihoods, as recommended by our literature check (LITERATURE §g.10).

``--listwise`` adds consensus information that no single scorer has (LITERATURE §g.11, RL-2):
``nsrc`` = number of systems that proposed the candidate, and minimum-Bayes-risk selection
(Kumar & Byrne, 2004; Goel & Byrne, 2000): with p = softmax(score / T) over the clip's candidates,
the final score is  score - lambda * sum_j p_j d(candidate, c_j),  d = word edits + 0.1 char edits.
T and lambda are tuned with the other weights; lambda = 0 recovers the plain hybrid.

Two modes:
* default: random search of the weights per domain on dev (scripted: unseen-sentence clips);
  weights saved to ``artifacts/cands/<name>.weights.json``;
* ``--weights``: reuse saved weights (dev-mode weights for final-mode systems, whose training has
  seen the dev sentences); ``--retune-spont`` re-tunes spontaneous on dev_spont, whose speakers stay
  unseen in every mode.
"""

from __future__ import annotations

import json
import math
from multiprocessing import get_context

import kenlm
import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from qxp_asr.decode import ctc_loglik, load_posteriors
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import pseudo_pool

LN10 = math.log(10)
DOMAINS = {"scripted": ("dev_scripted", "itest_scripted"), "spont": ("dev_spont", "itest_spont")}
TEST_TYPE = {"scripted": "scripted", "spont": "spontaneous"}
_LM: dict[str, kenlm.Model] = {}


def _feats(args: tuple) -> list[tuple[float, float]]:
    lp, texts, char_lm = args
    lm = _LM.setdefault(char_lm, kenlm.Model(char_lm))
    return [
        (ctc_loglik(lp, t), lm.score(" ".join("|" if c == " " else c for c in t)) * LN10)
        for t in texts
    ]


def _ctc_only(args: tuple) -> list[float]:
    lp, texts = args
    return [ctc_loglik(lp, t) for t in texts]


def _risk_matrix(texts: list[str]) -> np.ndarray:
    words = [t.split() for t in texts]
    return np.array(
        [
            [
                Levenshtein.distance(wa, wb) + 0.1 * Levenshtein.distance(a, b)
                for wb, b in zip(words, texts, strict=True)
            ]
            for wa, a in zip(words, texts, strict=True)
        ]
    )


def build_features(
    cands: pl.DataFrame,
    clip_ids: list[str],
    post: str,
    char_lm: str,
    workers: int,
    listwise: bool = False,
    view_feature: bool = False,
    extra_post: str = "",
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray | None]]:
    """Per clip: candidate texts, a feature matrix [ctc, chr, len, lp_*..., (nsrc)] and, with
    ``listwise``, the pairwise candidate distance matrix used for MBR."""
    lp_cols = sorted(c for c in cands.columns if c.startswith("lp_"))
    by_clip = cands.filter(pl.col("clip_id").is_in(clip_ids)).partition_by("clip_id", as_dict=True)
    posts = load_posteriors(ARTIFACTS / "posteriors" / post, clip_ids)
    jobs, metas = [], []
    for cid, lp in zip(clip_ids, posts, strict=True):
        g = by_clip.get((cid,))
        if g is None:
            continue
        texts = g["text"].to_list()
        llm = g.select(lp_cols).fill_null(-1e4).to_numpy() if lp_cols else np.zeros((len(texts), 0))
        tail = np.zeros((len(texts), 0))
        if view_feature:  # number of perturbed views (or control samples) that gave this candidate
            aug = pl.element().str.contains("@aug|@smp").cast(pl.Int32)
            naug = g["sources"].list.eval(aug).list.sum().to_numpy().astype(float)
            tail = np.concatenate([tail, naug[:, None]], axis=1)
        if listwise:
            tail = np.concatenate([tail, g["sources"].list.len().to_numpy()[:, None]], axis=1)
        jobs.append((lp, texts, char_lm))
        metas.append((cid, texts, llm, tail))
    with get_context("fork").Pool(workers) as pool:
        out = pool.map(_feats, jobs, chunksize=8)
        if extra_post:  # a second, independently trained CTC model as an extra verifier
            post2 = load_posteriors(ARTIFACTS / "posteriors" / extra_post, [m[0] for m in metas])
            jobs2 = [(lp2, m[1]) for lp2, m in zip(post2, metas, strict=True)]
            out2 = pool.map(_ctc_only, jobs2, chunksize=8)
    feats = {}
    for k, ((cid, texts, lps, tail), f) in enumerate(zip(metas, out, strict=True)):
        extra = [np.array(out2[k])[:, None]] if extra_post else []
        llm = np.concatenate([lps, *extra, tail], axis=1)
        base = np.array([[c, ch, len(t)] for (c, ch), t in zip(f, texts, strict=True)])
        risk = _risk_matrix(texts) if listwise else None
        feats[cid] = (texts, np.concatenate([base, llm], axis=1), risk)
    return feats


def _scale_ctc(feats: dict, weight: float) -> dict:
    if weight == 1.0:
        return feats
    out = {}
    for cid, (texts, f, risk) in feats.items():
        g = f.copy()
        g[:, 0] *= weight
        out[cid] = (texts, g, risk)
    return out


def pick(feats: dict, ids: list[str], w: np.ndarray) -> list[str]:
    """argmax of ctc + w . [chr, len, lp_*] per clip ("" if the clip has no candidates).

    Listwise features (risk matrix present): the last two weights are log T and lambda of MBR."""
    out = []
    for cid in ids:
        if cid not in feats:
            out.append("")
            continue
        texts, f, risk = feats[cid]
        if risk is None:
            out.append(texts[int(np.argmax(f[:, 0] + f[:, 1:] @ w))])
            continue
        s = f[:, 0] + f[:, 1:] @ w[:-2]
        z = (s - s.max()) / np.exp(w[-2])
        p = np.exp(z) / np.exp(z).sum()
        out.append(texts[int(np.argmax(s - w[-1] * (risk @ p)))])
    return out


def label_pool(table: pl.DataFrame, ids: list[str], feats: dict, w: np.ndarray) -> pl.DataFrame:
    """Hybrid 1-best of unlabelled clips plus confidence signals for filtering pseudo-labels:
    ``margin`` = score gap to the runner-up, ``n_sources`` = systems that proposed the 1-best."""
    if "sources" not in table.columns:  # plain CTC N-best table
        table = table.with_columns(sources=pl.concat_list("source"))
    srcs = {(c, t): len(s) for c, t, s in table.select("clip_id", "text", "sources").iter_rows()}
    rows = []
    for cid in ids:
        if cid not in feats:
            continue
        texts, f, _ = feats[cid]
        score = f[:, 0] + f[:, 1:] @ w
        order = np.argsort(-score)
        margin = float(score[order[0]] - score[order[1]]) if len(texts) > 1 else math.inf
        best = texts[int(order[0])]
        rows.append({"clip_id": cid, "hyp": best, "margin": margin, "n_sources": srcs[(cid, best)]})
    return pl.DataFrame(rows)


def tune(
    feats: dict,
    dev: pl.DataFrame,
    n_w: int,
    trials: int,
    seed: int,
    n_listwise: int = 0,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Random search + local refinement of the weights; ``init`` (e.g. the plain hybrid's weights
    with the listwise ones at 0) is the starting point, so the result is never worse on dev."""
    rng = np.random.default_rng(seed)
    ids, refs = dev["clip_id"].to_list(), dev["text"].to_list()

    def obj(w: np.ndarray) -> float:
        s = wer_cer(refs, pick(feats, ids, w))
        return s["wer"] + 0.5 * s["cer"]

    best_w = np.zeros(n_w) if init is None else init.copy()
    best = obj(best_w)
    n_lp = n_w - 2 - n_listwise
    for _ in range(trials):  # global random search, then local refinement around the best
        parts = [rng.uniform(0, 1.5, 1), rng.uniform(-1, 3, 1), rng.uniform(0, 2, n_lp)]
        if n_listwise:  # nsrc, log T, lambda
            parts += [rng.uniform(0, 3, 1), rng.uniform(0, 5, 1), rng.uniform(0, 3, 1)]
        w = np.concatenate(parts)
        if (v := obj(w)) < best:
            best, best_w = v, w
    for scale in (0.3, 0.1, 0.03):
        for _ in range(trials // 4):
            w = best_w + rng.normal(0, scale, n_w)
            if (v := obj(w)) < best:
                best, best_w = v, w
    return best_w


def main(
    cands: str = typer.Option(..., help="scored candidate table from llm_candidates.py"),
    post: str = typer.Option(..., help="CTC posteriors used for the ctc feature"),
    name: str = typer.Option(...),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("all", help="'indomain' = provided data only, 'all' = + external"),
    weights: str = typer.Option("", help="weights json to reuse (skip tuning)"),
    retune_spont: bool = typer.Option(False),
    spont_tune_roles: str = typer.Option(
        "dev_spont", help="roles the spontaneous weights are tuned on (dev mode: +itest_spont)"
    ),
    listwise: bool = typer.Option(False, help="add nsrc + MBR (consensus) to the score"),
    view_feature: bool = typer.Option(
        False, help="add the number of agreeing perturbed-view decodes (llm_candidates --views)"
    ),
    extra_post: str = typer.Option("", help="posteriors of a second CTC model (extra feature)"),
    scorers: str = typer.Option("", help="keep only these lp_<adapter> scorers (default: all)"),
    bag: int = typer.Option(0, help="bagging: average weights over this many bootstrap tunings"),
    ctc_weight: float = typer.Option(
        1.0, help="weight of the CTC log-likelihood (0: CTC only proposes, never scores)"
    ),
    init_weights: str = typer.Option(
        "", help="weights json to start the search from (missing features start at 0)"
    ),
    trials: int = typer.Option(400),
    workers: int = typer.Option(4),
    seed: int = typer.Option(0),
    apply_pool: bool = typer.Option(
        False, help="label the unlabelled spontaneous pool with the saved spont --weights"
    ),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    if scorers:  # e.g. cross-fitting: drop scorers that were trained on the clips being labelled
        keep = set(scorers.split(","))
        table = table.drop([c for c in table.columns if c.startswith("lp_") and c[3:] not in keep])
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    listwise_names = ["nsrc", "mbr_log_t", "mbr_lambda"] if listwise else []
    view_names = ["naug"] if view_feature else []
    extra_names = ["ctc2"] if extra_post else []
    feat_names = ["chr", "len", *lp_cols, *extra_names, *view_names, *listwise_names]
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    saved = json.loads((ARTIFACTS / "cands" / weights).read_text()) if weights else {}
    init = json.loads((ARTIFACTS / "cands" / init_weights).read_text()) if init_weights else {}
    if apply_pool:  # expert iteration (LITERATURE §g.11, RL-1): the hybrid labels the pool
        ids = pseudo_pool(man).filter(pl.col("clip_id").is_in(table["clip_id"].unique()))
        ids = ids["clip_id"].to_list()
        feats = build_features(table, ids, post, char_lm, workers, False, view_feature)
        feats = _scale_ctc(feats, ctc_weight)
        w = np.array([saved["spont"][k] for k in feat_names])
        lab = label_pool(table, ids, feats, w)
        lab.write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
        print(lab.describe())
        return
    chosen, outputs, results = {}, [], []
    for domain, (dev_role, itest_role) in DOMAINS.items():
        extra = ["ood"] if domain == "spont" else []
        apply = man.filter(
            pl.col("role").is_in([dev_role, itest_role, *extra])
            | ((pl.col("role") == "test") & (pl.col("test_type") == TEST_TYPE[domain]))
        )
        feats = build_features(
            table,
            apply["clip_id"].to_list(),
            post,
            char_lm,
            workers,
            listwise,
            view_feature,
            extra_post,
        )
        feats = _scale_ctc(feats, ctc_weight)
        if saved and not (domain == "spont" and retune_spont):
            w = np.array([saved[domain][k] for k in feat_names])
        else:
            tune_roles = spont_tune_roles.split(",") if domain == "spont" else [dev_role]
            dev = apply.filter(pl.col("role").is_in(tune_roles))
            if domain == "scripted":
                dev = dev.filter(pl.col("unseen_sentence"))
            w0 = np.array([init[domain].get(k, 0.0) for k in feat_names]) if init else None
            if bag:  # average weights tuned on bootstrap resamples of dev (variance reduction)
                ws = [
                    tune(
                        feats,
                        dev.sample(fraction=1.0, with_replacement=True, seed=seed + b),
                        len(feat_names),
                        trials,
                        seed + b,
                        len(listwise_names),
                        w0,
                    )
                    for b in range(bag)
                ]
                w = np.mean(ws, axis=0)
            else:
                w = tune(feats, dev, len(feat_names), trials, seed, len(listwise_names), w0)
        chosen[domain] = dict(zip(feat_names, map(float, w), strict=True))
        print(f"[{domain}] weights {chosen[domain]}", flush=True)
        out = apply.select("clip_id", "role", "unseen_sentence", "text", "test_type").with_columns(
            hyp=pl.Series(pick(feats, apply["clip_id"].to_list(), w))
        )
        outputs.append(out)
        lab = out.filter(pl.col("text").is_not_null())
        results += [
            {"domain": domain, "role": r, "unseen": u}
            | wer_cer(g["text"].to_list(), g["hyp"].to_list())
            for (r, u), g in lab.group_by("role", "unseen_sentence")
        ]
    pl.concat(outputs).write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
    (ARTIFACTS / "cands" / f"{name}.weights.json").write_text(json.dumps(chosen, indent=2))
    with pl.Config(tbl_rows=30):
        print(pl.DataFrame(results).sort("role", "unseen"))


if __name__ == "__main__":
    typer.run(main)
