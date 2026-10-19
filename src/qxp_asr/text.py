"""Text normalization for Puno Quechua ASR targets, LM text and hypotheses.

Conventions follow the organizers' curated reference (``validated_sentences_curated.tsv``):
lowercase, no punctuation, single spaces, apostrophe kept as part of ejective letters
(k' p' q' t' ch'), ``ñ`` kept. Disfluency / noise annotations used in the spontaneous corpus
(``{aa}``, ``[ruido]``, ``(na)``, ``{No audible}``) are dropped.

The official scorer (``eval/omnilingual/compute_wer.py`` in the organizer repo) only lowercases and
strips ``? ! ¿ ¡ . ,`` before computing micro-averaged WER/CER with jiwer, so anything we emit must
already be in the curated form.
"""

from __future__ import annotations

import re
import unicodedata

APOSTROPHES = "’‘ʼʻ`´′"
_APOS_RE = re.compile(f"[{APOSTROPHES}]")
_TAG_RE = re.compile(r"\{[^}]*\}|\[[^\]]*\]|\([^)]*\)|<[^>]*>")
_KEEP_RE = re.compile(r"[^a-zñ' ]")
_SPACE_RE = re.compile(r"\s+")
# Apostrophe only survives after an ejective-capable consonant and before a letter.
_BAD_APOS_RE = re.compile(r"(?<![kpqth])'|'(?![a-zñ])")

# Official scorer punctuation (organizer compute_wer.py).
_SCORER_PUNCT = str.maketrans("", "", "?!¿¡.,")


def strip_accents(text: str) -> str:
    """Remove combining accents but keep ``ñ`` (NFD would otherwise split it into n + tilde)."""
    out = []
    for ch in unicodedata.normalize("NFC", text):
        if ch in "ñÑ":
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        out.append("".join(c for c in decomposed if unicodedata.category(c) != "Mn"))
    return "".join(out)


def drop_tags(text: str) -> str:
    """Remove bracketed disfluency / noise annotations, then any stray bracket characters."""
    return re.sub(r"[\[\]{}()<>]", " ", _TAG_RE.sub(" ", text))


def normalize(text: str | None, *, keep_accents: bool = False) -> str:
    """Map raw text to the canonical ASR target alphabet ``[a-zñ' ]``.

    ``keep_accents=True`` keeps accented vowels instead of folding them (the organizer's
    spontaneous gold references occasionally keep e.g. ``chá``); default folds them, which
    matches the scripted curated references exactly.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = drop_tags(text)
    text = _APOS_RE.sub("'", text).lower()
    if not keep_accents:
        text = strip_accents(text)
        text = _KEEP_RE.sub(" ", text)
    else:
        text = re.sub(r"[^a-zñáéíóúü' ]", " ", text)
    text = _BAD_APOS_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def scorer_normalize(text: str) -> str:
    """Exact replica of the organizer scorer's ``remove_punct`` (lowercase, drop ?!¿¡.,)."""
    return text.translate(_SCORER_PUNCT).lower()


ALPHABET: tuple[str, ...] = (*"abcdefghijklmnopqrstuvwxyz", "ñ", "'")
