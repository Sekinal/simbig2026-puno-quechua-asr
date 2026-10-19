"""Word-level verifier, step 2: learn P(word is correct) and combine candidates word by word.

Reads ``scripts/word_features.py`` output. A gradient-boosted classifier (scikit-learn
HistGradientBoosting; seconds on CPU) predicts, for every competing word of every confusion-network
slot, whether it is the reference word. Uses:

* verification quality — AUC of detecting errors in the hybrid 1-best (pivot) words;
* learned word-level combination — per slot keep the pivot word unless an alternative's predicted
  correctness exceeds the pivot's by ``delta`` (conservative switching).

Protocol: speaker-disjoint 2-fold cross-fitting between the dev roles and the itest roles (train on
one, predict the other); OOD is predicted by a model trained on both and is never used to choose
anything. ``delta`` is chosen on the cross-fitted dev+itest predictions.

    uv run python scripts/word_verifier.py --feats wordfeat_dev_in
"""

from __future__ import annotations

import numpy as np
import polars as pl
import typer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from qxp_asr.combine import EPS
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS

FOLDS = {"dev": ["dev_scripted", "dev_spont"], "itest": ["itest_scripted", "itest_spont"]}
DELTAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
NON_FEATURES = {"clip_id", "role", "slot", "word", "label"}


def fit(df: pl.DataFrame, cols: list[str]) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
    )
    return clf.fit(df.select(cols).to_numpy(), df["label"].to_numpy())


def combine(df: pl.DataFrame, delta: float) -> dict[str, str]:
    """Per slot: the pivot word unless an alternative beats it by ``delta``."""
    out: dict[str, list[tuple[int, str]]] = {}
    for (cid, slot), g in df.group_by("clip_id", "slot", maintain_order=True):
        p = g["p"].to_numpy()
        piv = int(np.argmax(g["in_pivot"].to_numpy()))
        best = int(np.argmax(p))
        pick = best if p[best] > p[piv] + delta else piv
        out.setdefault(cid, []).append((slot, g["word"][pick]))
    return {c: " ".join(w for _, w in sorted(v) if w != EPS) for c, v in out.items()}


def report(df: pl.DataFrame, clips: pl.DataFrame, delta: float) -> pl.DataFrame:
    hyp = combine(df, delta)
    rows = []
    for (role, unseen), g in clips.filter(pl.col("clip_id").is_in(list(hyp))).group_by(
        "role", "unseen_sentence"
    ):
        refs = g["ref"].to_list()
        piv = wer_cer(refs, g["pivot"].to_list())
        new = wer_cer(refs, [hyp[c] for c in g["clip_id"]])
        rows.append(
            {
                "role": role,
                "unseen": unseen,
                "hybrid": piv["wer"],
                "verified": new["wer"],
                "clips": g.height,
            }
        )
    return pl.DataFrame(rows).sort("role", "unseen")


def main(
    feats: str = typer.Option(...),
    test_like: bool = typer.Option(
        True, help="train/evaluate only on test-like clips: unseen sentences + spontaneous"
    ),
    dump: str = typer.Option("", help="write cross-fitted P(correct) per word to cands/<dump>"),
    apply: str = typer.Option(
        "", help="features of unlabelled clips: train on all labelled rows, predict these"
    ),
    out: str = typer.Option("", help="with --apply: output name in artifacts/cands"),
) -> None:
    df = pl.read_parquet(ARTIFACTS / "cands" / f"{feats}.parquet")
    clips = pl.read_parquet(ARTIFACTS / "cands" / f"{feats}_clips.parquet").join(
        pl.read_parquet(MANIFESTS / "all.parquet").select("clip_id", "unseen_sentence"),
        on="clip_id",
    )
    df = df.with_columns(pl.selectors.starts_with("src_").fill_null(0))
    if test_like:  # seen-sentence clips teach "prefer memorised prompts", useless on the test
        keep = clips.filter(pl.col("unseen_sentence") | ~pl.col("role").str.contains("scripted"))
        df = df.filter(pl.col("clip_id").is_in(keep["clip_id"].to_list()))
    if apply:  # e.g. flag words of pseudo-labels for human review (scripts/make_review.py)
        new = pl.read_parquet(ARTIFACTS / "cands" / f"{apply}.parquet")
        new = new.with_columns(pl.selectors.starts_with("src_").fill_null(0))
        cols = [c for c in df.columns if c not in NON_FEATURES and c in new.columns]
        clf = fit(df.filter(pl.col("label").is_not_null()), cols)
        pred = new.with_columns(p=clf.predict_proba(new.select(cols).to_numpy())[:, 1])
        pred.write_parquet(ARTIFACTS / "cands" / f"{out}.parquet")
        print(f"{pred.height} word slots scored -> {out}")
        return
    cols = [c for c in df.columns if c not in NON_FEATURES]
    lab = df.filter(pl.col("label").is_not_null())
    preds = []
    for name, roles in FOLDS.items():  # cross-fit: train on the other speaker set
        train = lab.filter(~pl.col("role").is_in(roles) & (pl.col("role") != "ood"))
        test = lab.filter(pl.col("role").is_in(roles))
        clf = fit(train, cols)
        preds.append(test.with_columns(p=clf.predict_proba(test.select(cols).to_numpy())[:, 1]))
        print(f"fold {name}: trained on {train.height} words", flush=True)
    xfit = pl.concat(preds)
    ood = lab.filter(pl.col("role") == "ood")
    clf_all = fit(lab.filter(pl.col("role") != "ood"), cols)
    ood = ood.with_columns(p=clf_all.predict_proba(ood.select(cols).to_numpy())[:, 1])

    if dump:  # per-word predictions (cross-fitted; OOD from the model trained on dev+itest)
        pl.concat([xfit, ood]).select(
            "clip_id", "role", "slot", "word", "in_pivot", "label", "p"
        ).write_parquet(ARTIFACTS / "cands" / f"{dump}.parquet")
    piv = xfit.filter(pl.col("in_pivot") == 1)
    auc = roc_auc_score(1 - piv["label"].to_numpy(), 1 - piv["p"].to_numpy())
    piv_o = ood.filter(pl.col("in_pivot") == 1)
    auc_o = roc_auc_score(1 - piv_o["label"].to_numpy(), 1 - piv_o["p"].to_numpy())
    print(f"error-detection AUC on hybrid 1-best words: dev+itest {auc:.3f}, OOD {auc_o:.3f}")

    # choose delta per domain on the cross-fitted dev+itest predictions; OOD (spont) only reports
    with pl.Config(tbl_rows=20):
        for spont, sub in (
            (0, xfit.filter(pl.col("spont") == 0)),
            (1, xfit.filter(pl.col("spont") == 1)),
        ):
            scores = {}
            for d in DELTAS:
                r = report(sub, clips, d)
                scores[d] = float((r["verified"] - r["hybrid"]).mean())
            best = min(scores, key=scores.get)
            print(f"spont={spont}: change per delta {scores}; chosen {best}")
            print(report(sub, clips, best))
            if spont:
                print(report(ood, clips, best))


if __name__ == "__main__":
    typer.run(main)
