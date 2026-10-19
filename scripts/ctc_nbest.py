"""CTC word-LM beam-search N-best lists -> candidate table for hybrid rescoring.

Output ``artifacts/cands/<name>.parquet``: one row per (clip_id, text) with ``source="ctc"`` and
its beam rank. Consumed by ``scripts/llm_candidates.py`` (adds LLM N-best + LLM scores) and
``scripts/joint_rescore2.py``.

Run inside ``envs/omni`` (CPU):
    uv run python scripts/ctc_nbest.py --post ctc1b_v3_soup_wise0.85 --name ctc_v3
"""

from __future__ import annotations

from multiprocessing import get_context

import polars as pl
import typer
from pyctcdecode import build_ctcdecoder

from qxp_asr.decode import load_posteriors
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import pseudo_pool
from qxp_asr.vocab import pyctcdecode_labels

ROLES = "dev_scripted,dev_spont,itest_scripted,itest_spont,ood,test"


def main(
    post: str = typer.Option(...),
    name: str = typer.Option(...),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("all", help="'indomain' = provided data only, 'all' = + external"),
    alpha: float = typer.Option(0.5),
    keep: int = typer.Option(16),
    roles: str = typer.Option(ROLES),
    pool: bool = typer.Option(False, help="unlabelled spontaneous pool instead of --roles"),
    workers: int = typer.Option(4),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    man = pseudo_pool(man) if pool else man.filter(pl.col("role").is_in(roles.split(",")))
    lm_dir = ARTIFACTS / "lm" / lm
    decoder = build_ctcdecoder(
        pyctcdecode_labels(),
        str(lm_dir / f"{lm_text}.word4.bin"),
        unigrams=(lm_dir / f"{lm_text}.unigrams.txt").read_text().split(),
        alpha=alpha,
        beta=0.0,
    )
    ids = man["clip_id"].to_list()
    lps = load_posteriors(ARTIFACTS / "posteriors" / post, ids)
    with get_context("fork").Pool(workers) as procs:
        beams = decoder.decode_beams_batch(procs, lps, beam_width=64)
    rows = []
    for cid, bs in zip(ids, beams, strict=True):
        seen: set[str] = set()
        for rank, b in enumerate(bs[:keep]):
            text = " ".join(b[0].split())
            if text and text not in seen:
                seen.add(text)
                rows.append({"clip_id": cid, "text": text, "source": "ctc", "rank": rank})
    out = ARTIFACTS / "cands" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    print(f"{len(rows)} candidates for {len(ids)} clips -> {out}")


if __name__ == "__main__":
    typer.run(main)
