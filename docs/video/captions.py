"""Subtitles for the talk video, in a band below the animation (never on top of it).

Builds out/captions.ass from the narration (script.md) and its timing (out/timing.json: scene
durations and the start of every sentence, from your Space taps). A sentence is shown from its
start to the next sentence's start. Long sentences are split at commas, semicolons or colons into
captions of at most two lines, timed by their share of characters.

The frame layout is done by ffmpeg in finish.sh: the animation is scaled to 87.5% (1680x945) at
the top, and the captions sit in the 135 px band below it, on the same background colour.

    uv run python captions.py
"""

from __future__ import annotations

import json
import re
import textwrap

from narration import OUT, REC_TAIL, scenes, sentences

MAX_LINE = 74  # characters per caption line at 40 px on a 1680 px wide band
MAX_CAP = 2 * MAX_LINE - 6


def chunks(sentence: str) -> list[str]:
    """Split a long sentence at clause boundaries into caption-sized pieces."""
    if len(sentence) <= MAX_CAP:
        return [sentence]
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > MAX_CAP:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    out.append(cur)
    final = []
    for c in out:  # a clause still too long: split at a word boundary
        final += textwrap.wrap(c, MAX_CAP) if len(c) > MAX_CAP else [c]
    return final


def ts(t: float) -> str:
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def main() -> None:
    timing = json.loads((OUT / "timing.json").read_text())
    events, t0 = [], 0.0
    for sid, _title, text in scenes():
        dur = timing["duration"][sid]
        sents = sentences(text)
        starts = [c["t"] for c in timing["cues"][sid]]
        speech_end = dur - REC_TAIL + 0.3
        for k, s in enumerate(sents):
            a = starts[k]
            b = starts[k + 1] - 0.08 if k + 1 < len(sents) else min(speech_end, dur - 0.2)
            pieces = chunks(s)
            total = sum(len(p) for p in pieces)
            t = a
            for p in pieces:
                d = (b - a) * len(p) / total
                lines = textwrap.wrap(p, MAX_LINE)
                events.append((t0 + t, t0 + t + d, r"\N".join(lines)))
                t += d
        t0 += dur
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Band,Inter 18pt Medium,40,&H001B1D1D,&H001B1D1D,&H00F2F6F7,&H00F2F6F7,0,0,0,0,100,100,0,0,1,0,0,2,140,140,{18},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = "\n".join(
        f"Dialogue: 0,{ts(a)},{ts(b)},Band,,0,0,0,,{{\\fad(120,120)}}{txt}" for a, b, txt in events
    )
    (OUT / "captions.ass").write_text(head + body + "\n")
    print(f"{len(events)} captions -> {OUT / 'captions.ass'}")


if __name__ == "__main__":
    main()
