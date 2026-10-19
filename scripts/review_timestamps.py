"""Add word timestamps to a review queue (``make_review.py``) by CTC forced alignment.

For each clip, the hybrid transcript shown to the reviewer is force-aligned to the CTC v3
log-posteriors (``torchaudio.functional.forced_align``; 20 ms frames); each word slot gets
``t0``/``t1`` in seconds, so the review page can highlight the word being played and play a single
flagged word. Slots without a word (possible missing words) get the end of the previous word.

Run inside ``envs/omni``:
    uv run python scripts/review_timestamps.py --review /workspace/simbig/review
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer
from torchaudio.functional import forced_align, merge_tokens

from qxp_asr.decode import load_posteriors
from qxp_asr.paths import ARTIFACTS
from qxp_asr.vocab import INDEX, encode

FRAME_S = 0.02  # wav2vec 2.0 frame rate (320-sample stride at 16 kHz)
SPACE_ID = INDEX[" "]
POSTERIORS = {"dev": "ctc1b_v3_soup_wise0.85", "pool": "pool_ctc1b_v3"}


def word_times(lp: np.ndarray, words: list[str]) -> list[tuple[float, float]]:
    """Start/end seconds of each word from a CTC forced alignment of ``' '.join(words)``."""
    ids = encode(" ".join(words))
    emission = torch.from_numpy(lp.astype(np.float32)).unsqueeze(0)
    path, scores = forced_align(emission, torch.tensor([ids], dtype=torch.int32), blank=0)
    spans = merge_tokens(path[0], scores[0].exp(), blank=0)  # one span per target token, in order
    # group character spans into words (space tokens separate words)
    out, cur = [], []
    for tok, span in zip(ids, spans, strict=True):
        if tok == SPACE_ID:
            if cur:
                out.append((round(cur[0].start * FRAME_S, 2), round(cur[-1].end * FRAME_S, 2)))
            cur = []
        else:
            cur.append(span)
    if cur:
        out.append((round(cur[0].start * FRAME_S, 2), round(cur[-1].end * FRAME_S, 2)))
    return out


def main(review: Path = typer.Option(...)) -> None:
    raw = (review / "data.js").read_text()
    items = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
    kind = json.loads((review / "key.json").read_text())
    for it in items:
        lp = load_posteriors(ARTIFACTS / "posteriors" / POSTERIORS[kind[it["id"]]], [it["id"]])[0]
        words = [s for s in it["slots"] if s["word"]]
        try:
            times = word_times(lp, [s["word"] for s in words])
        except RuntimeError:  # alignment impossible (very short audio): spread uniformly
            step = it["duration"] / max(1, len(words))
            times = [(round(i * step, 2), round((i + 1) * step, 2)) for i in range(len(words))]
        for s, (t0, t1) in zip(words, times, strict=True):
            s["t0"], s["t1"] = t0, t1
        last = 0.0
        for s in it["slots"]:
            if s["word"]:
                last = s["t1"]
            else:
                s["t0"] = s["t1"] = last
    (review / "data.js").write_text(
        "const ITEMS = " + json.dumps(items, ensure_ascii=False) + ";\n"
    )
    print(f"{len(items)} clips aligned")


if __name__ == "__main__":
    typer.run(main)
