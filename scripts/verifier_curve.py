"""How good must a reference-free verifier be? WER of selection vs verifier ranking accuracy.

Starting from the tuned hybrid score (our real verifier, about 71-78 % pairwise accuracy), we blend
in the oracle: per clip, s = (1 - alpha) * z(hybrid score) + alpha * z(-error), with z the
within-clip standardisation, for alpha in [0, 1]. Each alpha gives a verifier of known pairwise
ranking accuracy; we report its selection WER on the perturbed-view candidate pool. The curve maps
"verifier quality" to "achievable WER" and places our verifier on it.

Blending towards the oracle is optimistic: the remaining mistakes sit where the hybrid was already
right. A second, pessimistic verifier family adds noise to the oracle instead,
s = z(-error) + sigma * N(0, 1) (mean over draws), so its mistakes fall on random pairs; at equal
pairwise accuracy the two families bracket what a better real verifier could reach.

    uv run python scripts/verifier_curve.py --cands pool_dev_aug \
        --weights hybrid_dev_aug.weights.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from joint_rescore2 import build_features
from qxp_asr.metrics import official_normalize, wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.stats import edits

SETS = {
    "dev_unseen": ((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"), "scripted"),
    "itest_unseen": ((pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"), "scripted"),
    "dev_spont": (pl.col("role") == "dev_spont", "spont"),
    "ood": (pl.col("role") == "ood", "spont"),
}
ALPHAS = [round(a, 2) for a in np.linspace(0, 1, 11)]
SIGMAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
NOISE_DRAWS = 5


def z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def main(
    cands: str = typer.Option(...),
    weights: str = typer.Option(...),
    post: str = typer.Option("ctc1b_v3_soup_wise0.85"),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("indomain"),
    workers: int = typer.Option(8),
    tag: str = typer.Option("", help="suffix of the output files (default: the paper's curve)"),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    saved = json.loads((ARTIFACTS / "cands" / weights).read_text())
    view_feature = "naug" in saved["scripted"]
    names = ["chr", "len", *lp_cols, *(["naug"] if view_feature else [])]
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("text").is_not_null() & (pl.col("duration_s") <= 39)
    )
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    rows, ours = [], []
    for name, (cond, domain) in SETS.items():
        sub = man.filter(cond)
        feats = build_features(
            table, sub["clip_id"].to_list(), post, char_lm, workers, False, view_feature
        )
        w = np.array([saved[domain][k] for k in names])
        clips = []
        for cid, ref in sub.select("clip_id", "text").iter_rows():
            if cid not in feats:
                continue
            texts, f, _ = feats[cid]
            r = official_normalize(ref)
            err = np.array(
                [
                    Levenshtein.distance(r.split(), official_normalize(t).split())
                    + 1e-3 * Levenshtein.distance(r, official_normalize(t))
                    for t in texts
                ]
            )
            clips.append((ref, texts, z(f[:, 0] + f[:, 1:] @ w), z(-err), err))
        rng = np.random.default_rng(0)
        settings = [("blend", a, 0) for a in ALPHAS]
        settings += [("noisy oracle", sg, d) for sg in SIGMAS for d in range(NOISE_DRAWS)]
        draws = []
        for family, x, _ in settings:
            refs, picks, correct, pairs = [], [], 0.0, 0
            per_clip = []  # (correct pairs, pairs) of each clip, for the interval of our verifier
            for ref, texts, hyb, orc, err in clips:
                if family == "blend":
                    s = (1 - x) * hyb + x * orc
                else:
                    s = orc + x * rng.standard_normal(len(orc))
                refs.append(ref)
                picks.append(texts[int(np.argmax(s))])
                better = err[:, None] < err[None, :]
                c = float((s[:, None] > s[None, :])[better].sum())
                correct += c
                pairs += int(better.sum())
                per_clip.append((c, int(better.sum())))
            if family == "blend" and x == 0:  # our verifier: clip-bootstrap intervals
                ci = _intervals(refs, picks, np.array(per_clip, dtype=float))
                ours.append({"set": name} | ci)
            draws.append(
                {
                    "set": name,
                    "family": family,
                    "x": x,
                    "pair_acc": 100 * correct / max(1, pairs),
                    "wer": wer_cer(refs, picks)["wer"],
                }
            )
        rows += (
            pl.DataFrame(draws)
            .group_by("set", "family", "x", maintain_order=True)
            .agg(pl.col("pair_acc", "wer").mean().round(2))
            .to_dicts()
        )
        print(pl.DataFrame(rows).filter(pl.col("set") == name), flush=True)
    sfx = f"_{tag}" if tag else ""
    pl.DataFrame(rows).write_csv(f"reports/verifier_curve{sfx}.csv")
    pl.DataFrame(ours).write_csv(f"reports/verifier_ours_ci{sfx}.csv")
    print(pl.DataFrame(ours))


def _intervals(refs: list[str], picks: list[str], pc: np.ndarray, n_res: int = 10_000) -> dict:
    """95 % clip-bootstrap intervals of our verifier's pairwise accuracy and selection WER."""
    e, n = edits(refs, picks)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(n), (n_res, len(n)))
    acc = 100 * pc[idx, 0].sum(1) / np.maximum(pc[idx, 1].sum(1), 1)
    wer = 100 * e[idx].sum(1) / n[idx].sum(1)
    return {
        "pair_acc": round(100 * pc[:, 0].sum() / pc[:, 1].sum(), 2),
        "pair_acc_lo": round(float(np.percentile(acc, 2.5)), 2),
        "pair_acc_hi": round(float(np.percentile(acc, 97.5)), 2),
        "wer": round(100 * e.sum() / n.sum(), 2),
        "wer_lo": round(float(np.percentile(wer, 2.5)), 2),
        "wer_hi": round(float(np.percentile(wer, 97.5)), 2),
    }


if __name__ == "__main__":
    typer.run(main)
