"""Character vocabulary shared by every CTC system we train (enables posterior-level ensembling).

Index 0 is the CTC blank, index 1 the word delimiter (space). The remaining symbols are the Puno
Quechua target alphabet (``qxp_asr.text.ALPHABET``).
"""

from __future__ import annotations

from collections.abc import Sequence

from qxp_asr.text import ALPHABET

BLANK = "<blank>"
SPACE = " "
SYMBOLS: tuple[str, ...] = (BLANK, SPACE, *ALPHABET)
INDEX = {s: i for i, s in enumerate(SYMBOLS)}
BLANK_ID = 0


def encode(text: str) -> list[int]:
    """Encode normalized text; characters outside the alphabet are dropped."""
    return [INDEX[c] for c in text if c in INDEX and c != BLANK]


def decode(ids: Sequence[int]) -> str:
    """Decode already-collapsed, blank-free ids."""
    return "".join(SYMBOLS[i] for i in ids if i != BLANK_ID).strip()


def ctc_collapse(frame_ids: Sequence[int]) -> list[int]:
    out: list[int] = []
    prev = -1
    for i in frame_ids:
        if i not in {prev, BLANK_ID}:
            out.append(i)
        prev = i
    return out


def pyctcdecode_labels() -> list[str]:
    """Label list in the format pyctcdecode expects ("" for blank, " " for the delimiter)."""
    return ["", *SYMBOLS[1:]]
