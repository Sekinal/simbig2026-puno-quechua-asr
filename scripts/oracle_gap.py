"""Selector headroom of a pooled candidate table (gate G0 of reports/rl_experiment_plan.md).

For every labelled role (and seen/unseen sentence split) report

* the hybrid 1-best WER/CER (``--pred``),
* the 1-best of each single system (argmax of its own ``lp_*`` among the candidates it proposed),
* the oracle WER/CER of the pool (per clip, the candidate with the fewest word edits, ties broken
  by character edits) and of each source's own candidates,
* how often the hybrid disagrees with each single system (the signal available to expert
  iteration / distillation from the hybrid).

If the oracle is close to the hybrid, better *selection* (MBR features, a discriminative scorer)
cannot help much and only better *candidates* can.

    uv run python scripts/oracle_gap.py --cands pool_dev --pred hybrid_dev
"""

from __future__ import annotations

import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from qxp_asr.metrics import official_normalize, wer_cer
from qxp_asr.paths import ARTIFACTS

ROLES = ("dev_scripted", "dev_spont", "ood", "itest_scripted", "itest_spont")


def _edits(ref: str, hyp: str) -> tuple[int, int]:
    return Levenshtein.distance(ref.split(), hyp.split()), Levenshtein.distance(ref, hyp)


def _oracle(refs: dict[str, str], cands: pl.DataFrame) -> dict[str, str]:
    best: dict[str, tuple[tuple[int, int], str]] = {}
    for cid, text in cands.select("clip_id", "text").iter_rows():
        if cid in refs:
            e = _edits(refs[cid], official_normalize(text))
            if cid not in best or e < best[cid][0]:
                best[cid] = (e, text)
    return {c: t for c, (_, t) in best.items()}


def _single(cands: pl.DataFrame, src: str) -> dict[str, str]:
    col = "lp_ctc" if src == "ctc" else f"lp_{src}"
    own = cands.filter(pl.col("sources").list.contains(src))
    if col not in own.columns:  # CTC candidates come ranked: keep the first one per clip
        return dict(
            own.group_by("clip_id", maintain_order=True)
            .first()
            .select("clip_id", "text")
            .iter_rows()
        )
    top = own.sort(col, descending=True).group_by("clip_id").first()
    return dict(top.select("clip_id", "text").iter_rows())


def _score(refs: list[str], ids: list[str], choice: dict[str, str], tag: str) -> dict:
    s = wer_cer(refs, [choice.get(c, "") for c in ids])
    return {f"{tag}_wer": round(s["wer"], 2), f"{tag}_cer": round(s["cer"], 2)}


def main(
    cands: str = typer.Option(..., help="artifacts/cands/<name>.parquet"),
    pred: str = typer.Option(..., help="artifacts/preds/<name>.parquet (hybrid 1-best)"),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    hyb = pl.read_parquet(ARTIFACTS / "preds" / f"{pred}.parquet").filter(
        pl.col("text").is_not_null() & pl.col("role").is_in(ROLES)
    )
    sources = sorted({s for row in table["sources"].to_list() for s in row})
    systems = {s: _single(table, s) for s in sources}
    rows = []
    for (role, unseen), g in hyb.group_by("role", "unseen_sentence"):
        ids, refs = g["clip_id"].to_list(), g["text"].to_list()
        ref_by = {c: official_normalize(r) for c, r in zip(ids, refs, strict=True)}
        sub = table.filter(pl.col("clip_id").is_in(ids))
        hyps = dict(zip(ids, g["hyp"].to_list(), strict=True))
        row = {"role": role, "unseen": unseen, "n": len(ids)}

        row |= _score(refs, ids, hyps, "hybrid") | _score(refs, ids, _oracle(ref_by, sub), "oracle")
        for s in sources:
            own = sub.filter(pl.col("sources").list.contains(s))
            row |= _score(refs, ids, systems[s], s) | _score(
                refs, ids, _oracle(ref_by, own), f"orc_{s}"
            )
            agree = sum(
                official_normalize(systems[s].get(c, "")) == official_normalize(hyps[c])
                for c in ids
            )
            row[f"disagree_{s}"] = round(100 * (1 - agree / len(ids)), 1)
        row["cands_per_clip"] = round(len(sub) / len(ids), 1)
        rows.append(row)
    out = pl.DataFrame(rows).sort("role", "unseen")
    with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=250):
        for cols in (
            ["hybrid", "oracle", *sources],
            [f"orc_{s}" for s in sources],
        ):
            print(out.select("role", "unseen", "n", *[f"{c}_wer" for c in cols]))
        print(out.select("role", "unseen", "cands_per_clip", *[f"disagree_{s}" for s in sources]))


if __name__ == "__main__":
    typer.run(main)
