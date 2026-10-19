"""Round-trip check of synthetic speech: transcribe ``artifacts/tts/<name>`` with our CTC model.

If our ASR transcribes the synthetic Cusco-Quechua voice about as well as human Puno recordings of
the same sentences, the synthetic audio is intelligible and in-distribution enough to serve as
back-translated training data; a much higher error means mismatch (voice, prosody, dialect).

Run inside ``envs/omni``:
    uv run python scripts/tts_roundtrip.py --name dev_unseen \
        --checkpoint /workspace/simbig/artifacts/merged/ctc1b_v3_soup_wise0.85.pt
"""

from __future__ import annotations

import polars as pl
import typer

from qxp_asr.metrics import wer_cer
from qxp_asr.omni import load_qxp_ctc, run_ctc
from qxp_asr.paths import ARTIFACTS
from qxp_asr.vocab import decode


def main(
    name: str = typer.Option(...),
    checkpoint: str | None = typer.Option(None),
    card: str = typer.Option("omniASR_CTC_1B_v2"),
    batch_seconds: float = typer.Option(160.0),
    device: str = typer.Option("cuda", help="cpu when the GPU is busy with training"),
) -> None:
    man = pl.read_parquet(ARTIFACTS / "tts" / name / "manifest.parquet")
    model = load_qxp_ctc(card, checkpoint, device)  # bf16, also on CPU (ctc_forward casts to it)
    items = list(man.select("clip_id", "wav", "duration_s").iter_rows())
    hyps = {
        o.clip_id: decode(o.greedy_ids.tolist())
        for o in run_ctc(model, items, batch_seconds, device)
    }
    pred = man.with_columns(hyp=pl.col("clip_id").replace_strict(hyps))
    pred.write_parquet(ARTIFACTS / "tts" / name / "roundtrip.parquet")
    s = wer_cer(pred["text"].to_list(), pred["hyp"].to_list())
    print(f"synthetic {name}: WER {s['wer']:.2f} CER {s['cer']:.2f} ({s['n']} utterances)")
    for t, h in pred.select("text", "hyp").head(5).iter_rows():
        print(f"  ref: {t}\n  hyp: {h}")


if __name__ == "__main__":
    typer.run(main)
