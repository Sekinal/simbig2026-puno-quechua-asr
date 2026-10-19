"""Deterministic development protocol (speaker-disjoint, with an unseen-sentence probe).

Roles assigned to every manifest row:

``dev_scripted`` / ``itest_scripted``
    Organizer ``splits_cv26`` scripted dev / test clips (21 + 21 speakers, disjoint from train).
``dev_spont`` / ``itest_spont``
    Gold spontaneous clips of the speakers in the official CV spontaneous dev / test splits.
``holdout``
    Any other clip of a dev/itest speaker (never trained on before the final retrain).
``holdout_sent``
    Clips of *training* speakers whose sentence is in the held-out unseen-sentence pool. Excluded in
    ``dev`` mode; included in ``org`` mode, which reproduces the organizers' training condition
    (every sentence seen, dev/itest speakers unseen) for a like-for-like comparison on itest.
``train``
    Everything else (labelled or not — label filtering happens in the data loaders).
``test``
    The shared-task test set.
``ood``
    Organizer radio-drama clips (broadcast speech): out-of-domain generalization check, never
    used for training or tuning.

``unseen_sentence`` flags scripted dev/itest clips whose sentence never occurs in ``train``. This
is the condition that matters: zero-shot hypotheses show the shared-task test reads *new*
sentences (median normalized edit distance to the closest pool sentence 0.495, identical to our
held-out sentences, vs 0.028 for in-pool sentences; ``scripts/pool_overlap.py``). Scripted model
selection and LM tuning therefore use the unseen subset.

For the final submission models, ``holdout`` and the itest roles are folded back into training.
"""

from __future__ import annotations

import hashlib

import polars as pl

N_UNSEEN_SENTENCES = 400
SEED = 13


def assign_roles(man: pl.DataFrame, n_unseen: int = N_UNSEEN_SENTENCES) -> pl.DataFrame:
    scripted = pl.col("corpus") == "scripted"
    spont = pl.col("corpus") == "spontaneous"

    def speakers(cond: pl.Expr) -> list[str]:
        return man.filter(cond)["speaker"].drop_nulls().unique().to_list()

    sdev = speakers(scripted & (pl.col("org_split") == "dev"))
    stest = speakers(scripted & (pl.col("org_split") == "test"))
    pdev = speakers(spont & (pl.col("cv_split") == "dev"))
    ptest = speakers(spont & (pl.col("cv_split") == "test"))

    eval_sentences = (
        man.filter(scripted & pl.col("org_split").is_in(["dev", "test"]))
        .select("text")
        .unique()
        .sort("text")
        .sample(n=n_unseen, seed=SEED)["text"]
        .to_list()
    )
    unseen = scripted & pl.col("text").is_in(eval_sentences)
    gold = pl.col("label").is_in(["gold", "pending"])

    role = (
        pl.when(pl.col("corpus") == "test")
        .then(pl.lit("test"))
        .when(pl.col("corpus") == "ood")
        .then(pl.lit("ood"))
        .when(scripted & (pl.col("org_split") == "dev"))
        .then(pl.lit("dev_scripted"))
        .when(scripted & (pl.col("org_split") == "test"))
        .then(pl.lit("itest_scripted"))
        .when(spont & gold & pl.col("speaker").is_in(pdev))
        .then(pl.lit("dev_spont"))
        .when(spont & gold & pl.col("speaker").is_in(ptest))
        .then(pl.lit("itest_spont"))
        .when(pl.col("speaker").is_in(sdev + stest + pdev + ptest))
        .then(pl.lit("holdout"))
        .when(unseen)
        .then(pl.lit("holdout_sent"))
        .otherwise(pl.lit("train"))
    )
    return man.with_columns(role=role, unseen_sentence=unseen & ~pl.col("text").is_null())


def stable_fold(key: str | None, salt: str, k: int = 2) -> int:
    """Deterministic fold of a speaker / sentence id (md5, independent of Python's hash seed)."""
    if not key:
        return -1
    return int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest(), 16) % k


def with_crossfit_folds(man: pl.DataFrame) -> pl.DataFrame:
    """Add ``spk_fold`` and ``sent_fold`` (2 folds each) for cross-fitting (LITERATURE §g.12).

    A fold-k model trains WITHOUT fold-k speakers and WITHOUT fold-k sentences
    (``crossfit_excluded``); it decodes gold clips of fold-k speakers whose sentence is in fold k
    (spontaneous clips have unique sentences, so only the speaker matters) — ``crossfit_heldout``.
    Its hypotheses then carry the errors of a model that has heard neither the voice nor the text,
    unlike decoding the training set with the full model (memorised, ~no errors).
    """
    return man.with_columns(
        spk_fold=pl.col("speaker").map_elements(
            lambda s: stable_fold(s, "spk"), return_dtype=pl.Int64
        ),
        sent_fold=pl.col("sentence_id").map_elements(
            lambda s: stable_fold(s, "sent"), return_dtype=pl.Int64
        ),
    )


def crossfit_excluded(k: int) -> pl.Expr:
    return (pl.col("spk_fold") == k) | (
        (pl.col("corpus") == "scripted") & (pl.col("sent_fold") == k)
    )


def crossfit_heldout(k: int) -> pl.Expr:
    return (pl.col("spk_fold") == k) & (
        (pl.col("corpus") == "spontaneous") | (pl.col("sent_fold") == k)
    )
