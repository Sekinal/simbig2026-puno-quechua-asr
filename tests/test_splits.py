"""Cross-fit folds (qxp_asr.splits)."""

import polars as pl

from qxp_asr.splits import crossfit_excluded, crossfit_heldout, stable_fold, with_crossfit_folds


def test_stable_fold_is_deterministic_and_balanced():
    folds = [stable_fold(f"spk{i}", "spk") for i in range(1000)]
    assert folds == [stable_fold(f"spk{i}", "spk") for i in range(1000)]
    assert 400 < sum(folds) < 600
    assert stable_fold(None, "spk") == -1


def test_heldout_clips_are_never_trained_on():
    man = with_crossfit_folds(
        pl.DataFrame(
            {
                "speaker": [f"s{i % 37}" for i in range(500)],
                "sentence_id": [f"t{i % 53}" for i in range(500)],
                "corpus": ["scripted"] * 400 + ["spontaneous"] * 100,
            }
        )
    )
    for k in (0, 1):
        train = man.filter(~crossfit_excluded(k))
        held = man.filter(crossfit_heldout(k))
        assert held.height > 0
        assert not set(train["speaker"]) & set(held["speaker"])
        scripted = held.filter(pl.col("corpus") == "scripted")
        train_scripted = train.filter(pl.col("corpus") == "scripted")
        assert not set(train_scripted["sentence_id"]) & set(scripted["sentence_id"])
