"""Run a (fine-tuned) qxp CTC model over manifest roles; save posteriors + greedy hypotheses.

Posteriors are float16 log-probabilities over ``qxp_asr.vocab.SYMBOLS`` (30 classes, 20 ms frames)
saved as ``artifacts/posteriors/<name>/<clip_id>.npy``; they feed LM decoding and ensembling.

Run inside ``envs/omni``:
    uv run python /workspace/simbig/repo/scripts/infer_ctc.py --name ctc1b_v1 \
        --checkpoint /workspace/simbig/artifacts/runs/ctc1b_v1/best.pt
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import typer

from qxp_asr.metrics import wer_cer
from qxp_asr.omni import load_qxp_ctc, run_ctc
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.sources import pseudo_pool
from qxp_asr.vocab import decode

DEFAULT_ROLES = "dev_scripted,dev_spont,itest_scripted,itest_spont,ood,test"


def report(pred: pl.DataFrame) -> pl.DataFrame:
    lab = pred.filter(pl.col("text").is_not_null())
    rows = [
        {"role": role, "unseen": unseen} | wer_cer(g["text"].to_list(), g["hyp"].to_list())
        for (role, unseen), g in lab.group_by("role", "unseen_sentence", maintain_order=True)
    ]
    return pl.DataFrame(rows).sort("role", "unseen") if rows else pl.DataFrame()


def main(
    name: str = typer.Option(...),
    checkpoint: str | None = typer.Option(None),
    card: str = typer.Option("omniASR_CTC_1B_v2"),
    roles: str = typer.Option(DEFAULT_ROLES),
    pool: bool = typer.Option(False, help="unlabelled spontaneous pool instead of --roles"),
    batch_seconds: float = typer.Option(160.0),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    man = pseudo_pool(man) if pool else man.filter(pl.col("role").is_in(roles.split(",")))
    items = list(man.select("clip_id", "wav", "duration_s").iter_rows())
    model = load_qxp_ctc(card, checkpoint)
    post_dir = ARTIFACTS / "posteriors" / name
    post_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hyps: dict[str, str] = {}
    for out in run_ctc(model, items, batch_seconds):
        hyps[out.clip_id] = decode(out.greedy_ids.tolist())
        np.save(post_dir / f"{out.clip_id}.npy", out.log_probs)
    rtfx = man["duration_s"].sum() / (time.time() - t0)
    print(f"{len(hyps)} clips (RTFx {rtfx:.0f}) -> {post_dir}")
    pred = man.select("clip_id", "role", "unseen_sentence", "text", "test_type").join(
        pl.DataFrame({"clip_id": list(hyps), "hyp": list(hyps.values())}), on="clip_id"
    )
    (ARTIFACTS / "preds").mkdir(parents=True, exist_ok=True)
    pred.write_parquet(ARTIFACTS / "preds" / f"{name}_greedy.parquet")
    with pl.Config(tbl_rows=30):
        print(report(pred))


if __name__ == "__main__":
    typer.run(main)
