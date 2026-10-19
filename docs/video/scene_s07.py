"""S07: the question. Encoder and decoder, two LoRA switches, what LoRA is, three options."""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    Arrow,
    Create,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    VGroup,
    VMobject,
)
from style import BG, CARD, DEC, ENC, GRID, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt


def waveform(width=1.0, height=0.7, seed=3, n=60, color=INK, stroke=2.2) -> VMobject:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    env = sum(
        a * np.exp(-(((x - c) / w) ** 2))
        for a, c, w in ((1.0, 0.22, 0.1), (0.7, 0.5, 0.08), (0.9, 0.76, 0.1))
    )
    env = 0.12 + env / env.max()
    y = env * np.sign(np.sin(np.arange(n) * 2.1)) * rng.uniform(0.45, 1.0, n)
    w = VMobject(stroke_color=color, stroke_width=stroke)
    w.set_points_as_corners(
        [[(xi - 0.5) * width, yi * height / 2, 0] for xi, yi in zip(x, y, strict=False)]
    )
    return w


def block(kind, label, color, w=3.2, h=1.5) -> VGroup:
    r = RoundedRectangle(width=w, height=h, corner_radius=0.2, stroke_color=color, stroke_width=3,
                         fill_color=color, fill_opacity=0.12)  # fmt: skip
    k, t = txt(kind, size=20, color=INK2), txt(label, size=32, weight=SEMIBOLD)
    k.move_to(r).align_to(
        r.get_center() + 0.5 * UP, UP
    )  # fixed top / baseline, so both blocks match
    t.move_to(r).align_to(r.get_center() + 0.38 * DOWN, DOWN)
    return VGroup(r, VGroup(k, t))


class Switch(VGroup):
    """A small toggle with a 'LoRA' label."""

    def __init__(self, color):
        pill = RoundedRectangle(width=0.9, height=0.42, corner_radius=0.21, stroke_color=REF, stroke_width=2,
                                fill_color=CARD, fill_opacity=1)  # fmt: skip
        knob = RoundedRectangle(width=0.32, height=0.32, corner_radius=0.16, stroke_width=0, fill_color=REF,
                                fill_opacity=1).move_to(pill.get_left() + 0.23 * RIGHT)  # fmt: skip
        label = txt("LoRA", size=22, color=INK2).next_to(pill, LEFT, buff=0.2)
        super().__init__(label, pill, knob)
        self.label, self.pill, self.knob, self.on_color = label, pill, knob, color

    def toggle(self, on: bool):
        """Animations that move this switch to the given state."""
        inset = self.pill.width * 0.255  # knob centre from the pill's end, at any scale
        x = self.pill.get_right() + inset * LEFT if on else self.pill.get_left() + inset * RIGHT
        return [
            self.pill.animate.set_fill(self.on_color if on else CARD, opacity=1).set_stroke(
                self.on_color if on else REF
            ),
            self.knob.animate.move_to(x).set_fill(BG if on else REF),
            self.label.animate.set_color(INK if on else INK2),
        ]


class S07(TalkScene):
    sid = "S07"

    def construct(self) -> None:
        self.intro_chip()

        # ---- encoder -> decoder ------------------------------------------------------------------
        y = 0.7
        enc = block("speech", "encoder", ENC).move_to([-2.1, y, 0])
        dec = block("LLM", "decoder", DEC).move_to([2.1, y, 0])
        wave = waveform().move_to([-5.25, y, 0])
        text = VGroup(*[
            RoundedRectangle(width=w, height=0.09, corner_radius=0.045, stroke_width=0, fill_color=INK,
                             fill_opacity=1)
            for w in (0.95, 0.75, 0.85)
        ]).arrange(DOWN, buff=0.13, aligned_edge=LEFT).move_to([5.25, y, 0])  # fmt: skip

        def arrow(a, b):
            return Arrow(a, b, buff=0.12, stroke_width=3, stroke_color=INK2, tip_length=0.16,
                         max_tip_length_to_length_ratio=0.3)  # fmt: skip

        a_in = arrow(wave.get_right(), enc.get_left())
        a_mid = arrow(enc.get_right(), dec.get_left())
        a_out = arrow(dec.get_right(), text.get_left())

        self.at(0)
        self.play(Create(wave), run_time=0.9, rate_func=SMOOTH)
        self.play(LaggedStart(FadeIn(a_in), FadeIn(enc, shift=0.15 * RIGHT), FadeIn(a_mid),
                              FadeIn(dec, shift=0.15 * RIGHT), FadeIn(a_out),
                              LaggedStart(*[GrowFromEdge(t, LEFT) for t in text], lag_ratio=0.2),
                              lag_ratio=0.2), run_time=2.2, rate_func=SMOOTH)  # fmt: skip

        # ---- what each part does ---------------------------------------------------------------
        self.at(1)

        def caption(*lines):
            return VGroup(*[txt(t, size=20, color=INK2) for t in lines]).arrange(DOWN, buff=0.06)

        dec_c = caption("a language model,", "already writes text").next_to(dec, DOWN, buff=0.22)
        enc_c = caption("turns sound into", "representations").next_to(enc, DOWN, buff=0.22)
        self.play(FadeIn(dec_c, shift=0.1 * UP), run_time=0.9, rate_func=SMOOTH)
        self.at(1, 3.6)
        self.play(FadeIn(enc_c, shift=0.1 * UP), run_time=0.9, rate_func=SMOOTH)

        # ---- the two switches ------------------------------------------------------------------
        self.at(2)
        s_enc, s_dec = Switch(ENC), Switch(DEC)
        s_enc.move_to([-2.1, y - 1.95, 0])
        s_dec.move_to([2.1, y - 1.95, 0])
        self.play(LaggedStart(FadeIn(s_enc, shift=0.1 * UP), FadeIn(s_dec, shift=0.1 * UP), lag_ratio=0.25),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip
        common = txt("the common recipe", size=18, color=MUTED).next_to(dec, UP, buff=0.22)
        self.play(
            *s_dec.toggle(True), FadeIn(common, shift=0.1 * DOWN), run_time=1.0, rate_func=SMOOTH
        )

        self.at(3, 0.4)
        recent = txt("recent work: hard, domain-shifted speech", size=18, color=MUTED).next_to(
            enc, UP, buff=0.22
        )
        self.play(
            *s_enc.toggle(True), FadeIn(recent, shift=0.1 * DOWN), run_time=1.0, rate_func=SMOOTH
        )

        # ---- LoRA: frozen W plus a thin low-rank B x A --------------------------------------------
        self.at(4)
        diagram = VGroup(wave, a_in, enc, a_mid, dec, a_out, text, enc_c, dec_c, s_enc, s_dec)
        self.play(FadeOut(common), FadeOut(recent), *s_enc.toggle(False), *s_dec.toggle(False),
                  run_time=0.6, rate_func=SMOOTH)  # fmt: skip
        home = diagram.get_center()
        self.play(diagram.animate.scale(0.62).move_to([0, 1.75, 0]), run_time=1.2, rate_func=SMOOTH)

        self.at(5)
        cy, d, r = -1.25, 2.0, 0.3
        W = Rectangle(
            width=d, height=d, stroke_color=REF, stroke_width=2, fill_color=CARD, fill_opacity=1
        )
        W.move_to([-2.6, cy, 0])
        cells = VGroup(*[
            Line(W.get_corner(DOWN + LEFT) + [k * d / 6, 0, 0], W.get_corner(UP + LEFT) + [k * d / 6, 0, 0],
                 stroke_color=GRID, stroke_width=1)
            for k in range(1, 6)
        ], *[
            Line(W.get_corner(DOWN + LEFT) + [0, k * d / 6, 0], W.get_corner(DOWN + RIGHT) + [0, k * d / 6, 0],
                 stroke_color=GRID, stroke_width=1)
            for k in range(1, 6)
        ])  # fmt: skip
        W_t = txt("W", size=40, weight=SEMIBOLD, color=INK2).move_to(W)
        W_l = txt("frozen weight", size=20, color=INK2).next_to(W, DOWN, buff=0.22)
        plus = txt("+", size=44, color=INK2).move_to([-0.95, cy, 0])
        B = Rectangle(width=r, height=d, stroke_width=0, fill_color=INK, fill_opacity=1)
        B.move_to([0.05, cy, 0])
        A = Rectangle(width=d, height=r, stroke_width=0, fill_color=INK, fill_opacity=1)
        A.next_to(B, RIGHT, buff=0.18).align_to(B, UP)
        B_t = txt("B", size=24, weight=SEMIBOLD).next_to(B, LEFT, buff=0.12).align_to(B, DOWN)
        A_t = txt("A", size=24, weight=SEMIBOLD).next_to(A, DOWN, buff=0.12)
        r_t = txt("rank r", size=18, color=MUTED).next_to(A, RIGHT, buff=0.18)
        BA_l = txt("small low-rank correction, trained", size=20, color=INK2)
        BA_l.next_to(VGroup(B, A), DOWN, buff=0.22).align_to(W_l, DOWN)
        self.play(FadeIn(W), Create(cells), FadeIn(W_t), run_time=1.2, rate_func=SMOOTH)
        self.play(FadeIn(W_l, shift=0.1 * UP), run_time=0.7, rate_func=SMOOTH)
        self.at(5, 2.6)
        self.play(FadeIn(plus), GrowFromEdge(B, UP), run_time=1.0, rate_func=SMOOTH)
        self.play(GrowFromEdge(A, LEFT), FadeIn(B_t), FadeIn(A_t), run_time=1.0, rate_func=SMOOTH)
        self.play(
            FadeIn(r_t, shift=0.1 * LEFT),
            FadeIn(BA_l, shift=0.1 * UP),
            run_time=0.9,
            rate_func=SMOOTH,
        )

        # "choose exactly which part learns": the correction goes to one place or the other
        self.at(5, 6.0)
        hl = VGroup(*[SurroundingRectangle(s, corner_radius=0.1, buff=0.08, stroke_width=2, stroke_color=c)
                      for s, c in ((s_enc, ENC), (s_dec, DEC))])  # fmt: skip
        self.play(FadeIn(hl), run_time=0.8, rate_func=SMOOTH)
        self.play(*s_enc.toggle(True), run_time=0.7, rate_func=SMOOTH)
        self.play(*s_enc.toggle(False), *s_dec.toggle(True), run_time=0.7, rate_func=SMOOTH)
        self.play(*s_dec.toggle(False), FadeOut(hl), run_time=0.7, rate_func=SMOOTH)

        # ---- the question --------------------------------------------------------------------------
        self.at(6)
        lora = VGroup(W, cells, W_t, W_l, plus, B, A, B_t, A_t, r_t, BA_l)
        self.play(FadeOut(lora, shift=0.15 * DOWN), run_time=0.8, rate_func=SMOOTH)
        self.play(diagram.animate.scale(1 / 0.62).move_to(home), run_time=1.2, rate_func=SMOOTH)

        self.at(7)
        opts = VGroup()
        for name in ("decoder only", "encoder only", "both"):
            t = txt(name, size=22, color=INK2, t2c={"decoder": DEC, "encoder": ENC})
            p = RoundedRectangle(width=2.5, height=0.55, corner_radius=0.27, stroke_color=REF, stroke_width=1.5,
                                 fill_color=CARD, fill_opacity=1)  # fmt: skip
            opts.add(VGroup(p, t.move_to(p)))
        opts.arrange(RIGHT, buff=0.35).move_to([0, -2.4, 0])
        same = txt("same data  ·  same training budget", size=18, color=MUTED).next_to(
            opts, DOWN, buff=0.25
        )
        self.play(LaggedStart(*[FadeIn(o, shift=0.1 * UP) for o in opts], lag_ratio=0.2), run_time=1.2,
                  rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(same, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        def pick(i):
            return [o[0].animate.set_stroke(INK if k == i else REF, width=2.5 if k == i else 1.5)
                    for k, o in enumerate(opts)]  # fmt: skip

        self.at(8, 0.2)
        self.play(*s_dec.toggle(True), *pick(0), run_time=0.8, rate_func=SMOOTH)
        self.at(8, 1.2)
        self.play(
            *s_dec.toggle(False), *s_enc.toggle(True), *pick(1), run_time=0.8, rate_func=SMOOTH
        )
        self.at(8, 2.3)
        self.play(*s_dec.toggle(True), *pick(2), run_time=0.8, rate_func=SMOOTH)

        q = txt("?", size=72, weight=SEMIBOLD).move_to(a_mid)
        self.play(FadeOut(a_mid), FadeIn(q, scale=0.7), run_time=1.0, rate_func=SMOOTH)
        self.end()
