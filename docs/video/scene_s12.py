"""S12: ranking, not coverage. Same best-in-beam, better ordering; then the fourth scorer."""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    Arrow,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    ValueTracker,
    VGroup,
    always_redraw,
)
from style import CARD, DEC, ENC, GRID, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt

# Oracle WER, best of 8 beams, held-out sentences (reports/mechanism_coverage.csv, itest_unseen v2 / ed).
ORACLE = (14.0, 14.0)
# Pairwise ranking accuracy on the shared pool (reports/mechanism_pairs.csv, read, all).
PAIR = (69.7, 76.9, "+7.3")
# Gain by kind of contrast (mechanism_pairs.csv, read): (label, delta, lo, hi, significant).
KINDS = (
    ("differ in several words", 8.2, 3.99, 13.39, True),
    ("differ in one letter: ejective", 6.93, -2.13, 18.12, False),
)
# Hybrid on held-out sentences without / with the encoder-adapted scorer (verifier_ours_ci*.csv).
HYB = (16.9, 15.5)

ROW1, ROW2 = 0.85, -0.1


def row_label(s: str, y: float) -> VGroup:
    return txt(s, size=22, color=INK).move_to([-4.05, y, 0], aligned_edge=RIGHT)


class S12(TalkScene):
    sid = "S12"

    def construct(self) -> None:
        self.intro_chip()

        lab1 = row_label("decoder LoRA", ROW1)
        lab2 = row_label("+ encoder LoRA", ROW2)
        divider = Line([0.15, 2.4, 0], [0.15, -1.0, 0], stroke_color=GRID, stroke_width=2)
        self.at(0)
        self.play(LaggedStart(FadeIn(lab1, shift=0.1 * RIGHT), FadeIn(lab2, shift=0.1 * RIGHT), lag_ratio=0.3),
                  run_time=1.1, rate_func=SMOOTH)  # fmt: skip

        # ---- left: coverage ---------------------------------------------------------------------
        lx0, lscale = -3.8, 0.17  # 14 % -> 2.4 units
        lt = txt("Is the right answer in the beam?", size=26, weight=SEMIBOLD).move_to(
            [-3.2, 2.35, 0]
        )
        ls = txt("best of 8 beams, WER (%)", size=20, color=MUTED).next_to(lt, DOWN, buff=0.14)
        self.at(1)
        self.play(
            FadeIn(lt, shift=0.1 * UP),
            FadeIn(ls, shift=0.1 * UP),
            Create(divider),
            run_time=0.9,
            rate_func=SMOOTH,
        )

        bars, vals = VGroup(), VGroup()
        for y, v, c in ((ROW1, ORACLE[0], DEC), (ROW2, ORACLE[1], ENC)):
            b = Rectangle(
                width=v * lscale, height=0.5, stroke_width=0, fill_color=c, fill_opacity=0.9
            )
            b.move_to([lx0, y, 0], aligned_edge=LEFT)
            bars.add(b)
            vals.add(txt(f"{v:.1f}", size=24, color=INK).next_to(b, RIGHT, buff=0.15))
        laxis = Line(
            [lx0, ROW1 + 0.45, 0], [lx0, ROW2 - 0.45, 0], stroke_color=INK2, stroke_width=1.5
        )
        self.at(2)
        self.play(Create(laxis), run_time=0.4)
        self.play(
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.35),
            run_time=1.5,
            rate_func=SMOOTH,
        )
        self.play(
            LaggedStart(*[FadeIn(v, shift=0.08 * RIGHT) for v in vals], lag_ratio=0.3), run_time=0.7
        )
        xe = lx0 + ORACLE[0] * lscale
        same = DashedLine(
            [xe, ROW1 + 0.4, 0],
            [xe, ROW2 - 0.4, 0],
            dash_length=0.07,
            stroke_color=INK2,
            stroke_width=2,
        )
        same_t = txt("the same", size=22, color=INK2).next_to(same, DOWN, buff=0.14)
        self.at(2, 3.6)
        self.play(Create(same), FadeIn(same_t, shift=0.08 * UP), run_time=0.9, rate_func=SMOOTH)
        left = VGroup(lt, ls, laxis, bars, vals, same, same_t)

        # ---- right: ranking ---------------------------------------------------------------------
        a0, a1, p0, p1 = 1.0, 6.0, 60.0, 80.0

        def ax(p: float) -> float:
            return a0 + (p - p0) / (p1 - p0) * (a1 - a0)

        rt = txt("Does the model rank it first?", size=26, weight=SEMIBOLD).move_to([3.5, 2.35, 0])
        rs = txt("pairs of candidates ordered correctly (%)", size=20, color=MUTED).next_to(
            rt, DOWN, buff=0.14
        )
        base = -0.75
        raxis = Line([a0 - 0.1, base, 0], [a1 + 0.1, base, 0], stroke_color=INK2, stroke_width=1.5)
        rticks = VGroup()
        for p in (60, 70, 80):
            rticks.add(
                Line([ax(p), base, 0], [ax(p), base - 0.07, 0], stroke_color=INK2, stroke_width=1.5)
            )
            rticks.add(txt(str(p), size=18, color=MUTED).move_to([ax(p), base - 0.28, 0]))
        guides = VGroup(
            *[
                Line([a0 - 0.1, y, 0], [a1 + 0.1, y, 0], stroke_color=GRID, stroke_width=1.5)
                for y in (ROW1, ROW2)
            ]
        )
        self.at(3)
        self.play(
            left.animate.set_opacity(0.3),
            FadeIn(rt, shift=0.1 * UP), FadeIn(rs, shift=0.1 * UP),
            Create(raxis), FadeIn(rticks), Create(guides),
            run_time=1.1, rate_func=SMOOTH,
        )  # fmt: skip

        d1 = Dot([ax(PAIR[0]), ROW1, 0], radius=0.12, color=DEC)
        d2 = Dot([ax(PAIR[1]), ROW2, 0], radius=0.12, color=ENC)
        v1 = txt(f"{PAIR[0]:.1f}", size=22, color=INK).next_to(d1, UP, buff=0.14)
        v2 = txt(f"{PAIR[1]:.1f}", size=22, color=INK).next_to(d2, UP, buff=0.14)
        arrow = Arrow([ax(PAIR[0]), ROW2, 0], [ax(PAIR[1]), ROW2, 0], buff=0.14, stroke_width=4, color=INK2,
                      max_tip_length_to_length_ratio=0.2, tip_length=0.18)  # fmt: skip
        drop = DashedLine(
            d1.get_center(),
            [ax(PAIR[0]), ROW2, 0],
            dash_length=0.06,
            stroke_color=REF,
            stroke_width=2,
        )
        gain = txt(f"{PAIR[2]} points", size=26, weight=SEMIBOLD).next_to(arrow, DOWN, buff=0.14)
        self.at(4)
        self.play(
            FadeIn(d1, scale=0.5), FadeIn(v1, shift=0.08 * UP), run_time=0.8, rate_func=SMOOTH
        )
        self.play(Create(drop), run_time=0.6, rate_func=SMOOTH)
        self.play(GrowArrow(arrow), FadeIn(d2, scale=0.5), run_time=1.1, rate_func=SMOOTH)
        self.play(
            FadeIn(v2, shift=0.08 * UP),
            FadeIn(gain, shift=0.08 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        right = VGroup(rt, rs, raxis, rticks, guides, d1, d2, v1, v2, drop, arrow, gain)

        # ---- by kind of contrast ----------------------------------------------------------------
        k0, k1, q0, q1 = 1.0, 4.4, -5.0, 20.0

        def kx(q: float) -> float:
            return k0 + (q - q0) / (q1 - q0) * (k1 - k0)

        ky = (-1.85, -2.6)
        zero = DashedLine([kx(0), ky[0] + 0.4, 0], [kx(0), ky[1] - 0.3, 0], dash_length=0.06, stroke_color=INK2,
                          stroke_width=1.5)  # fmt: skip
        zero_t = txt("0", size=18, color=MUTED).next_to(zero, DOWN, buff=0.08)
        k_head = txt("gain, by how two candidates differ", size=18, color=MUTED)
        k_head.move_to([0.55, ky[0] + 0.5, 0], aligned_edge=RIGHT)
        kinds = VGroup()
        for (name, d, lo, hi, sig), y in zip(KINDS, ky, strict=False):
            c = ENC if sig else REF
            lab = txt(name, size=22, color=INK if sig else INK2).move_to(
                [0.55, y, 0], aligned_edge=RIGHT
            )
            ci = Line([kx(lo), y, 0], [kx(hi), y, 0], stroke_color=c, stroke_width=5)
            dot = Dot([kx(d), y, 0], radius=0.1, color=c)
            note = txt("clear" if sig else "not significant", size=20, color=INK2 if sig else MUTED)
            note.next_to(ci, RIGHT, buff=0.18)
            kinds.add(VGroup(lab, ci, dot, note))
        self.at(5)
        self.play(
            left.animate.set_opacity(0.15),
            Create(zero),
            FadeIn(zero_t),
            FadeIn(k_head),
            run_time=0.7,
            rate_func=SMOOTH,
        )
        self.play(FadeIn(kinds[0][0], shift=0.1 * RIGHT), Create(kinds[0][1]), FadeIn(kinds[0][2], scale=0.5),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(kinds[0][3]), run_time=0.5)
        self.at(6, 0.6)
        self.play(FadeIn(kinds[1][0], shift=0.1 * RIGHT), Create(kinds[1][1]), FadeIn(kinds[1][2], scale=0.5),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip
        self.at(6, 4.0)
        self.play(FadeIn(kinds[1][3]), run_time=0.6)

        # ---- the takeaway line ------------------------------------------------------------------
        self.at(7)
        self.play(FadeOut(VGroup(zero, zero_t, k_head, kinds), shift=0.1 * DOWN), left.animate.set_opacity(1),
                  run_time=0.9, rate_func=SMOOTH)  # fmt: skip
        line = txt("It re-weights what the encoder already knows", size=32, t2c={"re-weights": ENC})
        line.move_to([0, -2.3, 0])
        self.at(8)
        self.play(FadeIn(line, shift=0.15 * UP), run_time=1.1, rate_func=SMOOTH)

        # ---- a fourth scorer in the reranker ----------------------------------------------------
        self.at(9)
        self.play(
            FadeOut(VGroup(left, right, lab1, lab2, divider), shift=0.15 * UP),
            line.animate.scale(0.8).move_to([0, 2.45, 0]),
            run_time=1.0, rate_func=SMOOTH,
        )  # fmt: skip

        def scorer(name: str, color: str) -> VGroup:
            box = RoundedRectangle(corner_radius=0.1, width=3.9, height=0.56, stroke_width=0, fill_color=CARD,
                                   fill_opacity=1)  # fmt: skip
            mark = Dot(radius=0.09, color=color).move_to(box.get_left() + 0.3 * RIGHT)
            t = txt(name, size=22, color=INK).next_to(mark, RIGHT, buff=0.2)
            return VGroup(box, mark, t)

        sc = VGroup(
            scorer("CTC likelihood", REF),
            scorer("character LM", REF),
            scorer("LLM, decoder LoRA", DEC),
            scorer("LLM, + encoder LoRA", ENC),
        ).arrange(DOWN, buff=0.16)
        rr_title = txt("reranker", size=24, weight=SEMIBOLD)
        frame = RoundedRectangle(
            corner_radius=0.18, width=4.4, height=3.2, stroke_color=INK2, stroke_width=1.5
        )
        rr = VGroup(frame, rr_title, sc)
        frame.move_to([-2.9, -0.35, 0])
        sc.move_to(frame).shift(0.0 * DOWN)
        rr_title.next_to(frame, UP, buff=0.15).align_to(frame, LEFT)
        hl4 = RoundedRectangle(
            corner_radius=0.1, width=3.9, height=0.56, stroke_color=ENC, stroke_width=2
        ).move_to(sc[3])
        self.play(Create(frame), FadeIn(rr_title), LaggedStart(*[FadeIn(s, shift=0.1 * RIGHT) for s in sc[:3]],
                  lag_ratio=0.2), run_time=1.3, rate_func=SMOOTH)  # fmt: skip

        w = ValueTracker(HYB[0])
        wcap = txt("WER, new sentences (%)", size=22, color=INK2).move_to([2.95, 0.65, 0])
        wnum = always_redraw(
            lambda: txt(f"{w.get_value():.1f}", size=60, weight=SEMIBOLD).move_to([2.95, -0.25, 0])
        )
        self.play(FadeIn(wcap), FadeIn(wnum), run_time=0.8, rate_func=SMOOTH)
        self.at(9, 3.4)
        sc[3].shift(0.5 * RIGHT).set_opacity(0)
        self.play(sc[3].animate.shift(0.5 * LEFT).set_opacity(1), run_time=1.0, rate_func=SMOOTH)
        self.play(Create(hl4), w.animate.set_value(HYB[1]), run_time=1.6, rate_func=SMOOTH)
        delta = txt("−1.4 points", size=28, weight=SEMIBOLD, t2c={"−1.4": ENC}).next_to(
            wnum, DOWN, buff=0.3
        )
        sub = txt("used in our final submission", size=20, color=MUTED).next_to(
            delta, DOWN, buff=0.18
        )
        self.play(FadeIn(delta, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.at(9, 9.0)
        self.play(FadeIn(sub, shift=0.08 * UP), run_time=0.8, rate_func=SMOOTH)
        self.end()
