"""Corpus-level (micro-averaged) WER / CER, as computed by the shared-task scorer.

Test README: references and hypotheses are NFC-normalized, lowercased and stripped of punctuation;
the apostrophe and ``ñ`` are kept. The organizer repo scores with jiwer over whole lists
(total edits / total reference words), which is what we do here.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import jiwer

_SPACE = re.compile(r"\s+")


def official_normalize(text: str | None) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c for c in text)
    return _SPACE.sub(" ", text).strip()


def wer_cer(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
    """Micro WER/CER in percent. Empty references are dropped (jiwer cannot score them)."""
    pairs = [
        (official_normalize(r), official_normalize(h)) for r, h in zip(refs, hyps, strict=True)
    ]
    pairs = [(r, h) for r, h in pairs if r]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "wer": 100 * jiwer.wer(r, h),
        "cer": 100 * jiwer.cer(r, h),
        "n": len(pairs),
    }
