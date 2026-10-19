"""S01: title card."""

from __future__ import annotations

from manim import DOWN, LEFT, SEMIBOLD, UP, Create, FadeIn, LaggedStart, Line, VGroup
from style import DEC, ENC, INK2, MUTED, SMOOTH, TalkScene, txt


class S01(TalkScene):
    sid = "S01"

    def construct(self) -> None:
        kicker = txt("SIMBig 2026  ·  Shared Task 1", size=22, color=MUTED, weight=SEMIBOLD)
        l1 = txt("Where should a speech model", size=58, weight=SEMIBOLD)
        l2 = txt("learn a new language?", size=58, weight=SEMIBOLD, t2c={"language": ENC})
        title = VGroup(l1, l2).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        sub = txt("Puno Quechua speech recognition with provided data only", size=28, color=INK2)
        rule = Line(LEFT * 0.0, LEFT * -2.2, stroke_width=5, color=ENC)
        rule2 = Line(LEFT * 0.0, LEFT * -1.1, stroke_width=5, color=DEC)
        rules = VGroup(rule, rule2).arrange(DOWN * 0 + LEFT * -1, buff=0.12)
        who = txt("Irving Ernesto Quezada Ramírez", size=26, weight=SEMIBOLD)
        aff = txt("Independent Researcher  ·  research@irvingernesto.com", size=20, color=MUTED)
        block = VGroup(
            kicker, title, sub, rules, VGroup(who, aff).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        )
        block.arrange(DOWN, buff=0.42, aligned_edge=LEFT).move_to([-0.3, 0.1, 0])
        self.play(FadeIn(kicker, shift=0.2 * UP), run_time=0.8, rate_func=SMOOTH)
        self.play(LaggedStart(*[FadeIn(w, shift=0.25 * UP) for w in (l1, l2)], lag_ratio=0.35),
                  run_time=1.6, rate_func=SMOOTH)  # fmt: skip
        self.play(
            Create(rule),
            Create(rule2),
            FadeIn(sub, shift=0.15 * UP),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        self.play(FadeIn(block[-1], shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)
        self.end()
