"""S14: takeaways. Three cards, the practitioner advice, limitations, and thanks."""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    RoundedRectangle,
    VGroup,
)
from style import CARD, DEC, ENC, GRID, INK, INK2, LM4, MUTED, REF, SMOOTH, TalkScene, txt

CARD_W, CARD_H, CARD_Y = 3.95, 4.15, 0.72
CARD_X = (-4.2, 0.0, 4.2)


def icon_reuse() -> VGroup:
    """Slope chart in miniature: the memorizing system climbs past ours (seen -> new sentences)."""
    l, r, lo, hi = -0.75, 0.75, -0.4, 0.4
    vio = Line([l, lo, 0], [r, hi, 0], stroke_color=LM4, stroke_width=4)
    ink = Line([l, 0.05, 0], [r, -0.02, 0], stroke_color=INK, stroke_width=4)
    dots = VGroup(*[Dot(p, radius=0.07, color=c) for p, c in
                    ((vio.get_start(), LM4), (vio.get_end(), LM4), (ink.get_start(), INK), (ink.get_end(), INK))])  # fmt: skip
    axis = VGroup(
        Line([l, -0.55, 0], [l, 0.55, 0], stroke_color=REF, stroke_width=1.5),
        Line([r, -0.55, 0], [r, 0.55, 0], stroke_color=REF, stroke_width=1.5),
    )
    return VGroup(axis, vio, ink, dots)


def icon_where() -> VGroup:
    """Encoder block feeding a decoder block."""
    enc = RoundedRectangle(
        corner_radius=0.1, width=0.95, height=0.85, stroke_width=0, fill_color=ENC, fill_opacity=0.9
    )
    dec = RoundedRectangle(
        corner_radius=0.1, width=0.95, height=0.85, stroke_width=0, fill_color=DEC, fill_opacity=0.9
    )
    VGroup(enc, dec).arrange(RIGHT, buff=0.55)
    arr = Arrow(
        enc.get_right(), dec.get_left(), buff=0.06, stroke_width=3, color=INK2, tip_length=0.14
    )
    return VGroup(enc, arr, dec)


def icon_verify() -> VGroup:
    """Many candidates, one chosen."""
    cands = VGroup(*[Dot(radius=0.075, color=REF) for _ in range(5)]).arrange(DOWN, buff=0.12)
    pick = Dot(radius=0.1, color=INK)
    ring = Circle(radius=0.22, stroke_color=INK, stroke_width=2)
    chosen = VGroup(ring, pick)
    VGroup(cands, chosen).arrange(RIGHT, buff=0.95)
    pick.move_to(ring)
    fan = VGroup(
        *[Line(c.get_center(), ring.get_left(), stroke_color=REF, stroke_width=1.5, stroke_opacity=0.8).scale(0.86)
          for c in cands]
    )  # fmt: skip
    return VGroup(fan, cands, chosen)


def lines(s: str, size: float, buff: float = 0.1, **kw) -> VGroup:
    """Multi-line text, each line centred."""
    return VGroup(*[txt(t, size=size, **kw) for t in s.split("\n")]).arrange(DOWN, buff=buff)


def card(x: float, icon: VGroup, title: str, body: str) -> VGroup:
    bg = RoundedRectangle(corner_radius=0.22, width=CARD_W, height=CARD_H, stroke_width=0, fill_color=CARD,
                          fill_opacity=1).move_to([x, CARD_Y, 0])  # fmt: skip
    icon.move_to(bg.get_top() + 0.68 * DOWN)
    head = lines(title, 24, buff=0.08, weight=SEMIBOLD).move_to(
        bg.get_top() + 1.6 * DOWN, aligned_edge=UP
    )
    text = lines(body, 19, buff=0.08, color=INK2).next_to(head, DOWN, buff=0.2)
    return VGroup(bg, icon, head, text)


class S14(TalkScene):
    sid = "S14"

    def construct(self) -> None:
        self.intro_chip()

        cards = VGroup(
            card(CARD_X[0], icon_reuse(), "Evaluate on\nnew sentences", "Sentence reuse can reverse\nwhich method looks best"),
            card(CARD_X[1], icon_where(), "Where to adapt\ndepends on the model",
                 "Encoder: 7B and Whisper\nDecoder: smaller encoders"),
            card(CARD_X[2], icon_verify(), "Verification is\nthe bottleneck",
                 "Better candidates are\neasy to find; choosing\nthem is hard"),
        )  # fmt: skip
        extra = lines("Where it helps: a low-rank\nupdate in the lower layers\nimproves ranking, not coverage", 18,
                      buff=0.07, color=INK)  # fmt: skip
        extra.next_to(cards[1][3], DOWN, buff=0.16)

        # placeholders first, then each card fills in as it is spoken
        self.at(0)
        for c in cards:
            c[0].set_fill(opacity=0.45)
        self.play(LaggedStart(*[FadeIn(c[0], shift=0.15 * UP) for c in cards], lag_ratio=0.15),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip

        def fill(c: VGroup) -> None:
            self.play(c[0].animate.set_fill(opacity=1), FadeIn(c[1], scale=0.9), FadeIn(c[2], shift=0.1 * UP),
                      run_time=0.9, rate_func=SMOOTH)  # fmt: skip

        self.at(1)
        fill(cards[0])
        self.at(2)
        self.play(FadeIn(cards[0][3], shift=0.08 * UP), run_time=0.7, rate_func=SMOOTH)
        self.at(3)
        fill(cards[1])
        self.at(4)
        self.play(FadeIn(cards[1][3], shift=0.08 * UP), run_time=0.7, rate_func=SMOOTH)
        self.at(5)
        self.play(FadeIn(extra, shift=0.08 * UP), run_time=0.8, rate_func=SMOOTH)
        self.at(6)
        fill(cards[2])
        self.at(7)
        self.play(FadeIn(cards[2][3], shift=0.08 * UP), run_time=0.7, rate_func=SMOOTH)

        # practitioner advice, one tip under each card
        head = txt("For practitioners", size=18, color=MUTED, weight=SEMIBOLD).move_to(
            [-6.15, -1.82, 0], aligned_edge=LEFT
        )
        tips = VGroup(
            *[
                lines(s, 21, buff=0.08, color=INK).move_to([x, -2.5, 0])
                for s, x in zip(
                    ("Test on new sentences\nand new speakers", "Try the encoder,\nbut test it",
                     "Spend compute on verifiers,\nnot more candidates"),
                    CARD_X, strict=False,
                )
            ]
        )  # fmt: skip
        rule = Line([-6.15, -1.58, 0], [6.15, -1.58, 0], stroke_color=GRID, stroke_width=1.5)
        self.at(8)
        self.play(Create(rule), FadeIn(head, shift=0.08 * UP), run_time=0.8, rate_func=SMOOTH)
        for i, off in enumerate((0.0, 3.6, 8.4)):
            self.at(9, off)
            self.play(FadeIn(tips[i], shift=0.1 * UP), run_time=0.9, rate_func=SMOOTH)

        # limitations, in fine print
        lim = txt("Limitations:  single seeds for the smaller models  ·  4–6 spontaneous evaluation speakers  ·  "
                  "official test scores pending", size=17, color=MUTED).move_to([0, -3.28, 0])  # fmt: skip
        self.at(10)
        self.play(FadeIn(lim, shift=0.08 * UP), run_time=0.9, rate_func=SMOOTH)

        # the last, calm frame
        contact = txt("Code on GitHub  ·  research@irvingernesto.com", size=26, color=INK2).move_to(
            [0, -0.55, 0]
        )
        thanks = txt("Thank you", size=60, weight=SEMIBOLD).move_to([0, 0.55, 0])
        self.at(11)
        rest = [m for m in self.content()]
        self.play(*[FadeOut(m, shift=0.15 * UP) for m in rest], run_time=1.0, rate_func=SMOOTH)
        self.play(FadeIn(contact, shift=0.1 * UP), run_time=1.0, rate_func=SMOOTH)
        self.at(12, -0.2)
        self.play(FadeIn(thanks, shift=0.15 * UP), run_time=1.2, rate_func=SMOOTH)
        self.end(fade=1.0)
