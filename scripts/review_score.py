"""Score human review decisions (``review_tool.html`` export) and build corrected training labels.

* Dev clips (hidden references, never trained on): for each flagged word, is the reviewer's choice
  the reference word of that confusion-network slot (from ``word_features.py`` labels)? Compared
  with the hybrid's own word, and reported with the "can't tell" rate and seconds per word.
* Pool clips: the corrected transcripts are written to ``artifacts/pseudo/<out>.parquet``
  (clip_id, target) for training; "can't tell" keeps the hybrid word.

    uv run python scripts/review_score.py --decisions /workspace/simbig/review/decisions_1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

from qxp_asr.combine import EPS
from qxp_asr.paths import ARTIFACTS, MANIFESTS


def chosen_word(ans: dict, slot: dict) -> str | None:
    """The reviewer's word for a slot ('' = no word), or None for "can't tell"."""
    c = ans["choice"]
    if c == "__unsure__":
        return None
    if c == "__other__":
        return ans.get("other", "").strip() or None
    return c


def main(
    decisions: list[Path] = typer.Option(...),
    review: Path = typer.Option(Path("/workspace/simbig/review")),
    out: str = typer.Option("human_review"),
    details: bool = typer.Option(False, help="print every dev decision next to the reference"),
) -> None:
    raw = (review / "data.notimes.js").read_text() if (review / "data.notimes.js").exists() else ""
    raw = raw or (review / "data.js").read_text()
    items = {it["id"]: it for it in json.loads(raw[raw.index("[") : raw.rindex("]") + 1])}
    kind = json.loads((review / "key.json").read_text())
    answers: dict = {}
    for d in decisions:  # later exports override earlier ones
        for cid, rec in json.loads(d.read_text()).items():
            if cid.startswith("__"):
                continue
            answers[cid] = rec
    labels = pl.read_parquet(ARTIFACTS / "cands" / "wordver_dev_in.parquet")
    refs = dict(
        pl.read_parquet(MANIFESTS / "all.parquet")
        .select("clip_id", "text")
        .drop_nulls()
        .iter_rows()
    )
    rows, corrected = [], []
    for cid, rec in answers.items():
        it = items[cid]
        slots = {s["k"]: s for s in it["slots"]}
        for k, ans in rec["answers"].items():
            s = slots[int(k)]
            word = chosen_word(ans, s)
            row = {
                "clip_id": cid,
                "kind": kind[cid],
                "slot": int(k),
                "model": s["word"],
                "human": word,
                "unsure": word is None,
                "changed": word not in (None, s["word"]),
            }
            if kind[cid] == "dev":
                lab = labels.filter((pl.col("clip_id") == cid) & (pl.col("slot") == int(k)))
                truth = lab.filter(pl.col("label") == 1)["word"].to_list()
                truth = [("" if w == EPS else w) for w in truth]
                ref_words = set(refs.get(cid, "").split())
                row["model_ok"] = s["word"] in truth
                row["truth"] = "|".join(truth) or "(not among options)"
                row["options"] = ", ".join(s["alts"][:5])
                row["human_ok"] = (
                    None
                    if word is None
                    else (word in truth or (word not in s["alts"] and word in ref_words))
                )
            rows.append(row)
        if kind[cid] == "pool":
            words = []
            for s in it["slots"]:
                a = rec["answers"].get(str(s["k"]))
                w = chosen_word(a, s) if a else None
                words.append(s["word"] if w is None else w)
            corrected.append({"clip_id": cid, "target": " ".join(w for w in words if w)})
    df = pl.DataFrame(rows, infer_schema_length=None)
    secs = sum(r["seconds"] for r in answers.values())
    print(f"{len(answers)} clips, {df.height} flagged words, {secs / max(1, df.height):.1f} s/word")
    print(
        df.group_by("kind").agg(
            pl.len(), unsure=pl.col("unsure").mean(), changed=pl.col("changed").mean()
        )
    )
    dev = df.filter(pl.col("kind") == "dev")
    if details and not dev.is_empty():
        with pl.Config(tbl_rows=100, tbl_width_chars=220, fmt_str_lengths=60):
            print(dev.select("model", "human", "truth", "options"))
    if not dev.is_empty():
        decided = dev.filter(~pl.col("unsure"))
        human_acc = decided["human_ok"].mean() if decided.height else 0.0
        print(
            f"dev: {dev.height} flagged words; model word correct {dev['model_ok'].mean():.2f};"
            f" human correct when deciding {human_acc:.2f}"
            f" ({decided.height} decided); changes that fixed / broke a word:"
            f" {(decided['human_ok'] & ~decided['model_ok']).sum()} /"
            f" {(~decided['human_ok'] & decided['model_ok']).sum()}"
        )
    if corrected:
        (ARTIFACTS / "pseudo").mkdir(exist_ok=True)
        pl.DataFrame(corrected).write_parquet(ARTIFACTS / "pseudo" / f"{out}.parquet")
        print(f"{len(corrected)} corrected pool transcripts -> pseudo/{out}.parquet")


if __name__ == "__main__":
    typer.run(main)
