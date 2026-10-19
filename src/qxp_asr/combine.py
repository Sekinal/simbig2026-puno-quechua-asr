"""System combination of 1-best hypotheses.

* ``medoid`` — sentence-level minimum-Bayes-risk selection (Kumar & Byrne, 2004; Goel & Byrne,
  2000): return the hypothesis with the lowest weighted word edit distance to all others.
* ``rover`` — word-level ROVER (Fiscus, 1997, "A post-processing system to yield reduced word error
  rates: Recognizer Output Voting Error Reduction"): progressively align hypotheses into a
  confusion network (slots of competing words, ``""`` = deletion) and take a weighted vote per slot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein

EPS = ""


def medoid(hyps: Sequence[str], weights: Sequence[float] | None = None) -> str:
    weights = weights or [1.0] * len(hyps)
    toks = [h.split() for h in hyps]

    def risk(i: int) -> float:
        return sum(
            w * Levenshtein.distance(toks[i], toks[j]) for j, w in enumerate(weights) if j != i
        )

    return hyps[min(range(len(hyps)), key=risk)]


def _align(slots: list[dict[str, float]], words: list[str]) -> list[tuple[int | None, str]]:
    """Edit-distance alignment of ``words`` against a confusion network.

    Substitution cost is 0 if the word already competes in the slot, else 1; returns a list of
    (slot index or None for an inserted slot, word or EPS).
    """
    n, m = len(slots), len(words)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + (0.0 if EPS in slots[i - 1] else 1.0)
        back[i][0] = "del"
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + 1.0
        back[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (0.0 if words[j - 1] in slots[i - 1] else 1.0)
            dele = cost[i - 1][j] + (0.0 if EPS in slots[i - 1] else 1.0)
            ins = cost[i][j - 1] + 1.0
            cost[i][j], back[i][j] = min((sub, "sub"), (dele, "del"), (ins, "ins"))
    path: list[tuple[int | None, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "sub":
            path.append((i - 1, words[j - 1]))
            i, j = i - 1, j - 1
        elif move == "del":
            path.append((i - 1, EPS))
            i -= 1
        else:
            path.append((None, words[j - 1]))
            j -= 1
    return path[::-1]


def rover(hyps: Sequence[str], weights: Sequence[float] | None = None) -> str:
    """Weighted word-level ROVER; the first hypothesis is the alignment pivot (best first)."""
    weights = list(weights or [1.0] * len(hyps))
    slots: list[dict[str, float]] = [defaultdict(float, {w: weights[0]}) for w in hyps[0].split()]
    seen_weight = weights[0]
    for hyp, weight in zip(hyps[1:], weights[1:], strict=True):
        new_slots: list[dict[str, float]] = []
        for slot_idx, word in _align(slots, hyp.split()):
            # an inserted slot means every earlier system implicitly voted for EPS there
            slot = defaultdict(float, {EPS: seen_weight}) if slot_idx is None else slots[slot_idx]
            slot[word] += weight
            new_slots.append(slot)
        slots = new_slots
        seen_weight += weight
    out = [max(slot.items(), key=lambda kv: (kv[1], kv[0] != EPS))[0] for slot in slots]
    return " ".join(w for w in out if w)


def confusion_network(hyps: Sequence[str]) -> list[dict[str, set[int]]]:
    """Align ``hyps`` (pivot first) into a confusion network.

    Returns slots mapping each competing word (``EPS`` = deletion) to the indices of the
    hypotheses that put it there; every hypothesis occupies exactly one entry per slot. Same
    progressive alignment as ``rover``, but keeping who voted for what (features for a learned
    word-level verifier, scripts/word_verifier.py).
    """
    slots: list[dict[str, set[int]]] = [defaultdict(set, {w: {0}}) for w in hyps[0].split()]
    for k, hyp in enumerate(hyps[1:], start=1):
        presence = [{w: set(v) for w, v in s.items()} for s in slots]
        new_slots: list[dict[str, set[int]]] = []
        for slot_idx, word in _align([dict.fromkeys(s, 1.0) for s in presence], hyp.split()):
            if slot_idx is None:  # inserted slot: all earlier hypotheses implicitly deleted
                slot: dict[str, set[int]] = defaultdict(set, {EPS: set(range(k))})
            else:
                slot = defaultdict(set, presence[slot_idx])
            slot[word].add(k)
            new_slots.append(slot)
        # slots this hypothesis skipped entirely do not occur: _align covers every slot
        slots = new_slots
    return slots
