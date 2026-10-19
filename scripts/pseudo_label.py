"""Build pseudo-label tables from several systems' hypotheses with agreement filtering.

Self-training inspirations (docs/LITERATURE.md §c): noisy student training for ASR (Park et al.,
2020, arXiv:2005.09629), iterative pseudo-labeling (Xu et al., 2020, arXiv:2005.09267), and
agreement-based filtering of pseudo-labels from multiple systems (e.g. ROVER-style consensus,
Fiscus 1997). The organizers' own silver labels (omniASR-LLM-7B) took their spontaneous WER from
13.6 to 3.2 (largely on silver references), so better labels for the ~30 h of untranscribed
spontaneous audio — which contains most of the elderly speech — are our biggest lever.

Input: ``--pred name[:weight]`` prediction parquets in ``artifacts/preds`` with (clip_id, hyp);
the first one is the alignment pivot. Output: ``artifacts/pseudo/<name>.parquet`` with
(clip_id, target, agreement, n_systems) for clips whose mean pairwise WER to the consensus is
below ``--max-disagreement``.

``--hybrid <preds>`` instead takes the 1-best of the hybrid rescorer on the pool
(``joint_rescore2.py --apply-pool``; expert iteration, LITERATURE §g.11 RL-1) and keeps clips whose
score margin is above the ``--min-margin-quantile`` quantile, whose 1-best was proposed by at least
``--min-sources`` systems, and whose speaking rate (chars/s) is within 0.5-1.5x the median.
"""

from __future__ import annotations

import polars as pl
import typer
from rapidfuzz.distance import Levenshtein

from qxp_asr.combine import rover
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.text import normalize


def word_error(ref: str, hyp: str) -> float:
    r = ref.split()
    return Levenshtein.distance(r, hyp.split()) / max(1, len(r))


def _repeats(text: str, n: int = 5) -> bool:
    """True if some word n-gram occurs twice (looping); natural reduplication is shorter."""
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return len(grams) != len(set(grams))


def from_hybrid(
    man: pl.DataFrame, hybrid: str, min_margin_quantile: float, min_sources: int, min_words: int
) -> pl.DataFrame:
    lab = pl.read_parquet(ARTIFACTS / "preds" / f"{hybrid}.parquet").join(
        man.select("clip_id", "duration_s"), on="clip_id"
    )
    lab = lab.with_columns(
        target=pl.col("hyp").map_elements(normalize, return_dtype=pl.String),
    ).with_columns(rate=pl.col("target").str.len_chars() / pl.col("duration_s"))
    med, q = lab["rate"].median(), lab["margin"].quantile(min_margin_quantile)
    print(f"median rate {med:.1f} chars/s, margin threshold {q:.2f}")
    return lab.filter(
        (pl.col("margin") >= q)
        & (pl.col("n_sources") >= min_sources)
        & pl.col("rate").is_between(0.5 * med, 1.5 * med)
        & (pl.col("target").str.count_matches(" ") + 1 >= min_words)
        & ~pl.col("target").map_elements(_repeats, return_dtype=pl.Boolean)
    ).select("clip_id", "target", "margin", "n_sources", "duration_s")


def main(
    name: str = typer.Option(...),
    pred: list[str] = typer.Option(
        [], help="pred parquet name[:weight] or 'silver'; the first one is the ROVER pivot"
    ),
    hybrid: str = typer.Option("", help="hybrid pool labels (joint_rescore2 --apply-pool)"),
    min_margin_quantile: float = typer.Option(0.3),
    min_sources: int = typer.Option(2),
    compare: str = typer.Option("", help="earlier pseudo table: report how much the labels change"),
    roles: str = typer.Option("train,holdout"),
    max_disagreement: float = typer.Option(0.15),
    min_words: int = typer.Option(2),
    min_systems: int = typer.Option(2, help="minimum number of available voters per clip"),
    split_age: bool = typer.Option(
        True, help="also write <name>_old (50+) / <name>_rest so older voices can be up-weighted"
    ),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        (pl.col("corpus") == "spontaneous")
        & pl.col("role").is_in(roles.split(","))
        & ~pl.col("label").is_in(["gold", "pending"])
    )
    if hybrid:
        kept = from_hybrid(man, hybrid, min_margin_quantile, min_sources, min_words)
        _write(kept, man, name, split_age)
        if compare:
            old = pl.read_parquet(ARTIFACTS / "pseudo" / f"{compare}.parquet")
            both = kept.join(old.select("clip_id", old_target="target"), on="clip_id")
            err = sum(
                Levenshtein.distance(o.split(), n.split())
                for o, n in both.select("old_target", "target").iter_rows()
            ) / max(1, both["old_target"].str.count_matches(" ").sum() + both.height)
            changed = (both["old_target"] != both["target"]).mean()
            print(
                f"vs {compare}: {both.height} shared clips, {100 * changed:.1f}% changed,"
                f" WER(old -> new) {100 * err:.1f}%"
            )
        n_in = pl.read_parquet(ARTIFACTS / "preds" / f"{hybrid}.parquet").height
        print(
            f"hybrid labels: {n_in} clips -> kept {kept.height}"
            f" ({kept['duration_s'].sum() / 3600:.1f} h)"
        )
        return
    systems = []
    for spec in pred:
        stem, _, w = spec.partition(":")
        # "silver" = organizer omniASR-LLM-7B labels stored in the manifest
        src = (
            man.select("clip_id", hyp=pl.col("silver")).drop_nulls()
            if stem == "silver"
            else pl.read_parquet(ARTIFACTS / "preds" / f"{stem}.parquet")
        )
        tab = src.select(
            "clip_id", pl.col("hyp").map_elements(normalize, return_dtype=pl.String).alias(stem)
        )
        systems.append((stem, float(w or 1.0), tab))
    df = man.select("clip_id", "duration_s")
    for _, _, tab in systems:  # left joins: a clip may lack some voters (e.g. no silver label)
        df = df.join(tab, on="clip_id", how="left")
    names = [s for s, _, _ in systems]
    weights = [w for _, w, _ in systems]
    rows = []
    for rec in df.iter_rows(named=True):
        votes = [(rec[n], w) for n, w in zip(names, weights, strict=True) if rec[n] is not None]
        if len(votes) < min_systems:
            continue
        hyps, ws = [h for h, _ in votes], [w for _, w in votes]
        target = rover(hyps, ws)
        agreement = sum(w * word_error(target, h) for h, w in votes)
        rows.append(
            {
                "clip_id": rec["clip_id"],
                "target": target,
                "disagreement": agreement / sum(ws),
                "n_systems": len(votes),
            }
        )
    out = pl.DataFrame(rows).join(df.select("clip_id", "duration_s"), on="clip_id")
    kept = out.filter(
        (pl.col("disagreement") <= max_disagreement)
        & (pl.col("target").str.count_matches(" ") + 1 >= min_words)
    )
    _write(kept, man, name, split_age)
    hours_in, hours_kept = out["duration_s"].sum() / 3600, kept["duration_s"].sum() / 3600
    print(
        f"{len(names)} systems, {out.height} candidate clips ({hours_in:.1f} h)"
        f" -> kept {kept.height} ({hours_kept:.1f} h)"
    )
    print(out["disagreement"].describe())


def _write(kept: pl.DataFrame, man: pl.DataFrame, name: str, split_age: bool) -> None:
    (ARTIFACTS / "pseudo").mkdir(parents=True, exist_ok=True)
    kept.write_parquet(ARTIFACTS / "pseudo" / f"{name}.parquet")
    if split_age:  # every spontaneous test speaker is 60+ (reports/eda/FINDINGS.md §1)
        old_ages = ["fifties", "sixties", "seventies", "eighties", "nineties"]
        aged = kept.join(man.select("clip_id", "age"), on="clip_id", how="left")
        is_old = pl.col("age").is_in(old_ages)
        aged.filter(is_old).write_parquet(ARTIFACTS / "pseudo" / f"{name}_old.parquet")
        aged.filter(~is_old | pl.col("age").is_null()).write_parquet(
            ARTIFACTS / "pseudo" / f"{name}_rest.parquet"
        )
        print(f"old (50+): {aged.filter(is_old)['duration_s'].sum() / 3600:.1f} h")


if __name__ == "__main__":
    typer.run(main)
