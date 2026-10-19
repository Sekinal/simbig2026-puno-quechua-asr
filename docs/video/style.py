"""Shared look and timing for every scene of the talk video (Manim CE).

Visual language: warm paper background, ink text in Inter, the paper's palette with fixed meanings
(blue = decoder adaptation, orange = encoder adaptation, grey = zero-shot / reference, violet = the
sentence-memorizing 4-gram system, aqua = synthetic verifiers, ink = our system).

Continuity across scenes, so that cuts are invisible:
* a section chip (top left) carries over from the previous scene and morphs into this scene's chip;
* a thin progress line along the bottom grows through the whole talk;
* every scene starts and ends on the bare background plus those two elements, so the last frame of
  one scene equals the first frame of the next.

Timing: each scene lasts exactly as long as its narration (out/timing.json, from narration.py).
``self.at(i)`` waits until narration sentence i starts; ``self.end()`` clears the content and pads
the scene to its exact duration.
"""

from __future__ import annotations

import json
from pathlib import Path

from manim import (
    DOWN,
    DR,
    LEFT,
    NORMAL,
    RIGHT,
    SEMIBOLD,
    UL,
    UP,
    AnimationGroup,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Text,
    TransformMatchingShapes,
    VGroup,
    config,
    rate_functions,
)

HERE = Path(__file__).parent
if not (HERE / "out" / "timing.json").exists():
    raise SystemExit(
        "out/timing.json is missing: run `uv run python narration.py` first (see README)"
    )
TIMING = json.loads((HERE / "out" / "timing.json").read_text())
ORDER = list(TIMING["duration"])
TOTAL = sum(TIMING["duration"].values())

# ---- palette (identical to scripts/paper_figs/style.py) ------------------------------------------
BG = "#f7f6f2"
DEC = "#2a78d6"  # decoder adaptation
ENC = "#e8602c"  # encoder adaptation
ENC_RAMP = ("#eb8a52", "#e8602c", "#a63d12")  # encoder size 300M -> 1B -> 7B
REF = "#a8a7a2"  # zero-shot / reference
SYN = "#169c6c"  # synthetic verifiers
LM4 = "#5a48b5"  # word 4-gram on the training prompts
INK = "#1d1d1b"  # primary text, our system
INK2 = "#55544f"  # secondary text, axes
MUTED = "#8a8984"  # tertiary text
GRID = "#e4e3de"  # recessive lines
CARD = "#efeee9"  # card / panel fill
FONT = "Inter"

config.background_color = BG

CHIPS = {
    "S01": "",
    "S02": "The problem",
    "S03": "The sentence-reuse trap",
    "S04": "How we evaluate",
    "S05": "The system",
    "S06": "Results",
    "S07": "Where should a model learn?",
    "S08": "It depends on the model",
    "S09": "Same information, different plasticity",
    "S10": "Which layers matter",
    "S11": "How many directions",
    "S12": "Ranking, not coverage",
    "S13": "Verification is the bottleneck",
    "S14": "Takeaways",
    "S15": "References",
}

# On-screen sources, bottom right in fine print, shown for the whole scene (full list: S15).
SOURCES = {
    "S02": [
        "Speakers: 2017 census (mother tongue), via Huaman et al. (SIMBig 2025)  ·  Map: Natural Earth",
        "Data: Common Voice Scripted 27.0, Spontaneous 5.0 (Ardila et al. 2020)  ·  "
        "Quechua phonology: Adelaar & Muysken (2004)",
    ],
    "S03": [
        "Results: this paper, Sect. 2 (dev set; 95% interval: speaker × sentence bootstrap, "
        "Owen 2007)"
    ],
    "S04": [
        "Protocol: this paper, Sect. 2  ·  Bootstrap: Bisani & Ney (2004), Owen (2007)  ·  "
        "Holm (1979)"
    ],
    "S05": [
        "System: this paper, Sect. 3  ·  Omnilingual ASR team (2025)  ·  LoRA: Hu et al. (2022)"
        "  ·  ROVER: Fiscus (1997)"
    ],
    "S06": [
        "Results: this paper, Sect. 4, Table 1  ·  37 public checkpoints re-run by us  ·  "
        "Holm (1979)"
    ],
    "S07": ["Encoder adaptation: Shi et al., EAVA (IEEE SLT 2026)  ·  LoRA: Hu et al. (2022)"],
    "S08": [
        "Results: this paper, Sect. 5, Fig. 1  ·  Whisper: Radford et al. (2023)  ·  "
        "Omnilingual ASR team (2025)"
    ],
    "S09": [
        "Results: this paper, Sect. 5, Fig. 2  ·  Probes and control tasks: Hewitt & Liang (2019)"
        "  ·  CKA: Kornblith et al. (2019)"
    ],
    "S10": ["Results: this paper, Sect. 5, Fig. 3a"],
    "S11": [
        "Results: this paper, Sect. 5, Fig. 3b  ·  Intrinsic dimension: Aghajanyan et al. (2021)"
    ],
    "S12": ["Results: this paper, Sect. 5"],
    "S13": [
        "Results: this paper, Sect. 6  ·  GSPO: Zheng et al. (2025)  ·  MWER: Prabhavalkar et al."
        " (2018)  ·  Dr GRPO: Liu et al. (2025)"
    ],
    "S14": ["This paper: I. E. Quezada Ramírez, SIMBig 2026 Shared Task 1 system description"],
}

SMOOTH = rate_functions.ease_in_out_cubic  # default easing for moves and morphs


def txt(s: str, size: float = 30, color: str = INK, weight=NORMAL, **kw) -> Text:
    """Inter text. Sizes: 64-72 title, 40-48 headline, 28-32 body, 20-24 labels, 17+ fine print."""
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, **kw)


def chip(sid: str) -> VGroup:
    """Section label at the top left (empty for the title scene)."""
    label = CHIPS.get(sid, "")
    if not label:
        return VGroup()
    n = ORDER.index(sid)
    t = txt(f"{n:02d}  {label.upper()}", size=17, color=MUTED, weight=SEMIBOLD)
    return t.to_corner(UL, buff=0.45)


def sources(sid: str) -> VGroup:
    """Fine-print source lines at the bottom right, above the progress line."""
    lines = [txt(t, size=15, color=MUTED) for t in SOURCES.get(sid, [])]
    if not lines:
        return VGroup()
    g = VGroup(*lines).arrange(DOWN, buff=0.07, aligned_edge=RIGHT)
    return g.to_corner(DR, buff=0.0).shift(0.42 * LEFT + 0.2 * UP)


def progress_line(frac: float) -> Line:
    left = config.frame_width / 2 * LEFT + (config.frame_height / 2 - 0.06) * DOWN
    length = max(frac * config.frame_width, 0.001)
    return Line(left, left + length * RIGHT, stroke_width=4, color=ENC)


class TalkScene(Scene):
    """Base class: set ``sid``; write ``construct`` with ``self.at``; finish with ``self.end()``."""

    sid = "S00"

    def setup(self) -> None:
        self.dur = TIMING["duration"][self.sid]
        self.cues = [c["t"] for c in TIMING["cues"][self.sid]]
        i = ORDER.index(self.sid)
        start = sum(TIMING["duration"][s] for s in ORDER[:i])
        self.frac0, self.frac1 = start / TOTAL, (start + self.dur) / TOTAL
        self.bar = progress_line(self.frac0)
        prev = ORDER[i - 1] if i > 0 else None
        self.chip = chip(prev) if prev else VGroup()
        self.add(self.bar, self.chip)
        self.persistent = [self.bar]

    def now(self) -> float:
        return self.renderer.time

    def wait_until(self, t: float) -> None:
        if t - self.now() > 1 / config.frame_rate:
            self.wait(t - self.now())

    def at(self, i: int, offset: float = 0.0) -> None:
        """Wait until narration sentence i (0-based) starts, plus ``offset`` seconds."""
        self.wait_until(self.cues[i] + offset)

    def intro_chip(self, run_time: float = 0.9) -> None:
        """Morph the previous scene's chip into this scene's chip (call first in construct)."""
        new = chip(self.sid)
        src = sources(self.sid)  # part of the content: it fades out with the scene in end()
        extra = [FadeIn(src)] if len(src) else []
        if len(self.chip) and len(new):
            self.play(TransformMatchingShapes(self.chip, new), *extra, run_time=run_time,
                      rate_func=SMOOTH)  # fmt: skip
        elif len(new):
            self.play(FadeIn(new, shift=0.1 * DOWN), *extra, run_time=run_time, rate_func=SMOOTH)
        self.chip = new
        self.persistent.append(self.chip)

    def content(self) -> list:
        keep = {id(m) for m in self.persistent}
        return [m for m in self.mobjects if id(m) not in keep]

    def end(self, fade: float = 0.8) -> None:
        """Fade out all content, grow the progress line, and pad to the exact scene duration."""
        remaining = self.dur - self.now()
        fade = max(0.2, min(fade, remaining - 1 / config.frame_rate))
        self.wait_until(self.dur - fade)
        new_bar = progress_line(self.frac1)
        group = [FadeOut(m, shift=0.15 * UP) for m in self.content()]
        self.play(
            AnimationGroup(*group, self.bar.animate.become(new_bar)),
            run_time=max(self.dur - self.now(), 1 / config.frame_rate),
            rate_func=SMOOTH,
        )
