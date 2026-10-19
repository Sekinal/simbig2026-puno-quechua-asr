"""Build a human-review queue of verifier-flagged words (training-pool labels + dev control).

Human-in-the-loop correction of TRAINING labels (never the test set). For untranscribed pool
clips, elderly speakers first (all spontaneous test speakers are 60+), the word-level verifier
(``word_verifier.py --apply``) flags the least reliable words of the hybrid pseudo-label; each
flagged slot offers the competing words of the confusion network. Dev spontaneous clips with the
reference hidden are mixed in (shuffled, indistinguishable) to measure the reviewer's accuracy;
they are never trained on. Writes ``<out>/data.js`` (items for the local review page),
``<out>/key.json`` (which items are dev; not loaded by the page) and ``<out>/wavs.txt``.

    uv run python scripts/make_review.py --out /workspace/simbig/review
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
import typer

from qxp_asr.combine import EPS
from qxp_asr.paths import ARTIFACTS, MANIFESTS

OLD = {"sixties": 0, "seventies": 0, "eighties": 0, "nineties": 0, "fifties": 1}


def flagged_items(preds: str, feats: str, clip_ids: list[str], thr: float, per_clip: int) -> dict:
    p = pl.read_parquet(ARTIFACTS / "cands" / f"{preds}.parquet").filter(
        pl.col("clip_id").is_in(clip_ids)
    )
    clips = pl.read_parquet(ARTIFACTS / "cands" / f"{feats}_clips.parquet")
    pivot_text = dict(zip(clips["clip_id"], clips["pivot"], strict=True))
    out = {}
    for (cid,), g in p.group_by("clip_id"):
        slots = []
        for (k,), s in g.sort("slot").group_by("slot", maintain_order=True):
            piv = s.filter(pl.col("in_pivot") == 1)
            if piv.is_empty():
                continue
            alts = s.sort("p", descending=True)
            slots.append(
                {
                    "k": int(k),
                    "word": piv["word"][0],
                    "p": float(piv["p"][0]),
                    "alts": [w if w != EPS else "" for w in alts["word"].to_list()],
                }
            )
        uncertain = sorted(
            (sl for sl in slots if sl["p"] < thr and len(sl["alts"]) > 1), key=lambda sl: sl["p"]
        )[:per_clip]
        if not uncertain:
            continue
        chosen = {sl["k"] for sl in uncertain}
        for sl in slots:
            sl["flagged"] = sl["k"] in chosen
        out[cid] = {"slots": slots, "pivot": pivot_text.get(cid, "")}
    return out


def main(
    out: Path = typer.Option(...),
    n_pool: int = typer.Option(70),
    n_dev: int = typer.Option(30),
    thr: float = typer.Option(0.5, help="flag pivot words with P(correct) below this"),
    per_clip: int = typer.Option(4),
    seed: int = typer.Option(0),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("duration_s").is_between(3, 20))
    pool = man.filter(
        pl.col("clip_id").is_in(
            pl.read_parquet(ARTIFACTS / "cands" / "wordfeat_pool_clips.parquet")["clip_id"]
        )
    )
    pool_items = flagged_items(
        "wordver_pool", "wordfeat_pool", pool["clip_id"].to_list(), thr, per_clip
    )
    pool = pool.filter(pl.col("clip_id").is_in(list(pool_items))).with_columns(
        prio=pl.col("age").replace_strict(OLD, default=2, return_dtype=pl.Int64)
    )
    pool = pool.sample(fraction=1.0, shuffle=True, seed=seed).sort("prio").head(n_pool)
    dev = man.filter(pl.col("role") == "dev_spont")
    dev_items = flagged_items(
        "wordver_dev_in", "wordfeat_dev_in", dev["clip_id"].to_list(), thr, per_clip
    )
    dev = dev.filter(pl.col("clip_id").is_in(list(dev_items))).sample(
        n=min(n_dev, len(dev_items)), seed=seed
    )
    items = []
    for kind, df, src in (("pool", pool, pool_items), ("dev", dev, dev_items)):
        for cid, wav, dur, age in df.select("clip_id", "wav", "duration_s", "age").iter_rows():
            items.append(
                {"id": cid, "kind": kind, "wav": wav, "duration": round(dur, 1), "age": age}
                | src[cid]
            )
    random.Random(seed).shuffle(items)
    out.mkdir(parents=True, exist_ok=True)
    page = [
        {
            "id": it["id"],
            "audio": f"audio/{it['id']}.wav",
            "duration": it["duration"],
            "slots": it["slots"],
        }
        for it in items
    ]
    (out / "data.js").write_text("const ITEMS = " + json.dumps(page, ensure_ascii=False) + ";\n")
    (out / "key.json").write_text(json.dumps({it["id"]: it["kind"] for it in items}, indent=1))
    (out / "wavs.txt").write_text("\n".join(it["wav"] for it in items) + "\n")
    n_flag = sum(sl["flagged"] for it in items for sl in it["slots"])
    print(f"{len(items)} clips ({len(pool)} pool, {len(dev)} dev), {n_flag} flagged words")
    print(pool.group_by("age").len().sort("len", descending=True))


if __name__ == "__main__":
    typer.run(main)
