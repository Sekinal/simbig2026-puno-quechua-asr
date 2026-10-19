"""S06: results. OOD comparison against 37 public systems, then the held-out-sentence staircase."""

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
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    LaggedStart,
    Line,
    MoveToTarget,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
)
from style import CARD, DEC, ENC, GRID, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt

# OOD (radio drama) WER of the 37 public systems, sorted; reports/benchmarks/all_systems.md, rows 6-42.
PUBLIC = [
    20.85, 23.26, 23.49, 24.47, 24.70, 26.06, 30.89, 31.04, 31.19, 31.80, 32.85, 34.21, 34.97, 35.42, 38.22,
    38.37, 39.20, 39.35, 40.71, 45.09, 45.54, 46.00, 49.55, 50.30, 54.98, 60.35, 88.37, 97.13, 99.55, 100.00,
    100.00, 100.00, 101.51, 114.58, 136.78, 175.91, 1637.99,
]  # fmt: skip
ORG_BEST = 3  # index of the organizers' best checkpoint (xls-r-cpt-qxp-cv26-silver, 24.47)
OURS = 14.1  # submission, paper Section 4

# held-out sentences (itest unseen), paper Table 1
STAIRS = [
    ("zero-shot", 27.9, REF),
    ("decoder LoRA", 20.2, DEC),
    ("+ encoder", 16.0, ENC),
    ("full system", 15.5, INK),
]


def check(color=INK, s=0.18) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=4)
    m.set_points_as_corners([[-s, 0, 0], [-s / 3, -s * 0.75, 0], [s, s * 0.8, 0]])
    return m


class S06(TalkScene):
    sid = "S06"

    def construct(self) -> None:
        self.intro_chip()

        # ---- OOD comparison ------------------------------------------------------------------
        x0, x1, y0, h = -5.0, 6.0, -2.55, 4.4  # plot box: baseline y0, height h
        n = len(PUBLIC) + 1
        slot = (x1 - x0) / n
        bw = slot * 0.56
        ymax = ValueTracker(100.0)

        def ypos(v):
            return y0 + h * min(v, ymax.get_value()) / ymax.get_value()

        def bar(i, v, color):
            r = Rectangle(width=bw, height=max(ypos(v) - y0, 0.001), stroke_width=0, fill_color=color,
                          fill_opacity=1)  # fmt: skip
            return r.move_to([x0 + (i + 0.5) * slot, y0, 0], aligned_edge=DOWN)

        base = Line([x0, y0, 0], [x1, y0, 0], stroke_color=INK2, stroke_width=1.5)
        title = txt("Radio drama  ·  out of domain  ·  WER %", size=24, color=INK2).move_to(
            [0, 2.65, 0]
        )

        def grid():
            g = VGroup()
            m = ymax.get_value()
            step = 20 if m > 50 else 10
            for v in range(step, int(m) + 1, step):
                y = ypos(v)
                g.add(Line([x0, y, 0], [x1, y, 0], stroke_color=GRID, stroke_width=1))
                g.add(txt(f"{v}", size=17, color=MUTED).next_to([x1, y, 0], RIGHT, buff=0.15))
            return g

        grid_m = always_redraw(grid)
        bars = VGroup(*[bar(i + 1, v, REF) for i, v in enumerate(PUBLIC)])

        self.at(0)
        self.play(FadeIn(title, shift=0.1 * DOWN), Create(base), run_time=1.0, rate_func=SMOOTH)
        self.play(FadeIn(grid_m), run_time=0.8, rate_func=SMOOTH)
        big = VGroup(txt("ours", size=26, color=INK2), txt("14.1", size=64, weight=SEMIBOLD))
        big.arrange(RIGHT, buff=0.25, aligned_edge=DOWN).move_to([-3.7, 1.6, 0])
        self.play(FadeIn(big, shift=0.15 * UP), run_time=1.0, rate_func=SMOOTH)

        # 37 public systems grow from the baseline
        self.at(1)
        n_lbl = txt("37 public systems, sorted", size=20, color=INK2).next_to(base, DOWN, buff=0.3)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05), FadeIn(n_lbl),
                  run_time=2.4, rate_func=SMOOTH)  # fmt: skip
        # zoom the axis to 0-40 so the leaders are legible; bars above 40 are clipped
        live = always_redraw(lambda: VGroup(*[bar(i + 1, v, REF) for i, v in enumerate(PUBLIC)]))
        self.remove(*bars)
        self.add(live)
        self.play(ymax.animate.set_value(40.0), run_time=2.0, rate_func=SMOOTH)
        bars = VGroup(*[bar(i + 1, v, REF) for i, v in enumerate(PUBLIC)])
        grid_s = grid()
        self.remove(live, grid_m)
        self.add(grid_s, bars)
        clip_lbl = txt("clipped at 40", size=17, color=MUTED).next_to(
            [x1, y0, 0], DOWN, buff=0.3, aligned_edge=RIGHT
        )

        # ours slides in last, at the front of the line
        self.at(1, 5.0)
        ours = bar(0, OURS, INK)
        ours_l = VGroup(txt("ours", size=18, color=INK2), txt("14.1", size=24, weight=SEMIBOLD))
        ours_l.arrange(DOWN, buff=0.05, aligned_edge=RIGHT)
        ours_l.next_to(ours, LEFT, buff=0.2).align_to(ours, UP).shift(0.05 * DOWN)
        self.play(
            FadeIn(ours, shift=0.6 * RIGHT), Transform(big, ours_l), run_time=1.4, rate_func=SMOOTH
        )

        # best public 20.9, organizers' best 24.5
        best, org = bars[0], bars[ORG_BEST]
        ref_y = ypos(PUBLIC[0])
        ref = DashedLine(
            [x0, ref_y, 0], [x1, ref_y, 0], dash_length=0.08, stroke_color=INK2, stroke_width=1.5
        )
        best_l = VGroup(
            txt("best public", size=18, color=INK2), txt("20.9", size=22, weight=SEMIBOLD)
        )
        best_l.arrange(DOWN, buff=0.06).move_to([best.get_x() + 0.1, 1.4, 0])
        best_lead = Line(best_l.get_bottom() + 0.08 * DOWN, best.get_top() + 0.08 * UP, stroke_color=INK2,
                         stroke_width=1.2)  # fmt: skip
        org_l = VGroup(
            txt("organizers' best", size=18, color=INK2), txt("24.5", size=22, weight=SEMIBOLD)
        )
        org_l.arrange(DOWN, buff=0.06).move_to([org.get_x() + 1.1, 2.0, 0])
        org_lead = Line(org_l.get_bottom() + 0.08 * DOWN, org.get_top() + 0.08 * UP, stroke_color=INK2,
                        stroke_width=1.2)  # fmt: skip
        self.play(best.animate.set_fill(MUTED), Create(ref), FadeIn(best_l, shift=0.1 * DOWN), Create(best_lead),
                  FadeIn(clip_lbl), run_time=1.2, rate_func=SMOOTH)  # fmt: skip
        self.play(org.animate.set_fill(MUTED), FadeIn(org_l, shift=0.1 * DOWN), Create(org_lead),
                  run_time=1.2, rate_func=SMOOTH)  # fmt: skip

        self.at(2)
        gap = Arrow([ours.get_x(), ref_y, 0], [ours.get_x(), ypos(OURS) + 0.02, 0], buff=0, stroke_width=3,
                    stroke_color=INK, max_tip_length_to_length_ratio=0.4, tip_length=0.14)  # fmt: skip
        third = VGroup(
            txt("a third", size=18, weight=SEMIBOLD), txt("fewer errors", size=18, weight=SEMIBOLD)
        )
        third.arrange(DOWN, buff=0.05, aligned_edge=RIGHT).next_to(ours_l, UP, buff=0.3).align_to(
            ours_l, RIGHT
        )
        self.play(GrowArrow(gap), FadeIn(third, shift=0.1 * RIGHT), run_time=1.2, rate_func=SMOOTH)

        ood = VGroup(title, base, grid_s, bars, ref, best_l, best_lead, org_l, org_lead, clip_lbl, ours, big,
                     gap, third, n_lbl)  # fmt: skip

        # ---- confirmatory hypotheses ---------------------------------------------------------------
        self.at(3)
        self.play(FadeOut(ood, shift=0.15 * UP), run_time=1.0, rate_func=SMOOTH)

        def card(tag, line):
            r = RoundedRectangle(width=8.4, height=1.15, corner_radius=0.16, stroke_color=INK2, stroke_width=1.5,
                                 fill_color=CARD, fill_opacity=1)  # fmt: skip
            t = VGroup(txt(tag, size=24, weight=SEMIBOLD), txt(line, size=24))
            t.arrange(RIGHT, buff=0.4).move_to(r).align_to(r, LEFT).shift(0.4 * RIGHT)
            return VGroup(r, t)

        h_title = VGroup(
            txt("Two confirmatory hypotheses", size=30, weight=SEMIBOLD),
            txt("designated after the experiments", size=20, color=MUTED),
        ).arrange(DOWN, buff=0.12)
        h_title.move_to([0, 2.0, 0])
        h1 = card("H1", "full system beats its best component")
        h2 = card("H2", "perturbed views help")
        cards = VGroup(h1, h2).arrange(DOWN, buff=0.3).move_to([0, 0.15, 0])
        holm = txt("Holm-corrected for multiple testing", size=20, color=MUTED).next_to(
            cards, DOWN, buff=0.4
        )
        self.play(FadeIn(h_title, shift=0.1 * UP), run_time=1.0, rate_func=SMOOTH)
        self.play(LaggedStart(*[FadeIn(c, shift=0.15 * UP) for c in cards], lag_ratio=0.3), run_time=1.4,
                  rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(holm), run_time=0.8, rate_func=SMOOTH)

        self.at(4)
        self.play(h2.animate.set_opacity(0.25), run_time=0.9, rate_func=SMOOTH)
        self.at(5)
        ck = check().move_to(h1[0].get_right() + 0.5 * LEFT)
        hl = SurroundingRectangle(
            h1[0], corner_radius=0.18, buff=0.06, stroke_color=INK, stroke_width=2.5
        )
        self.play(Create(ck), Create(hl), run_time=1.0, rate_func=SMOOTH)
        sub = txt("on held-out sentences  ·  Holm-adjusted p = 0.04", size=20, color=INK2)
        sub.move_to(holm)
        self.play(
            FadeOut(holm, shift=0.1 * DOWN),
            FadeIn(sub, shift=0.1 * DOWN),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        ood_ns = txt("out of domain: p = 0.12 after correction (n.s.)", size=18, color=MUTED)
        ood_ns.next_to(sub, DOWN, buff=0.15)
        self.at(5, 4.6)
        self.play(FadeIn(ood_ns, shift=0.1 * DOWN), run_time=0.9, rate_func=SMOOTH)

        # ---- staircase on held-out sentences ---------------------------------------------------
        self.at(6, -0.8)
        self.play(
            FadeOut(VGroup(h_title, cards, ck, hl, sub, ood_ns), shift=0.15 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        s_title = txt("Held-out sentences  ·  WER %", size=24, color=INK2).move_to([0, 2.65, 0])
        sy0, sh, smax = -2.2, 4.0, 30.0
        xs = [-4.2, -1.4, 1.4, 4.2]
        sbw = 1.35

        def sy(v):
            return sy0 + sh * v / smax

        s_base = Line([-5.6, sy0, 0], [5.6, sy0, 0], stroke_color=INK2, stroke_width=1.5)
        self.play(FadeIn(s_title, shift=0.1 * DOWN), Create(s_base), run_time=1.0, rate_func=SMOOTH)

        s_bars, s_vals, s_names = VGroup(), VGroup(), VGroup()
        for (name, v, c), x in zip(STAIRS, xs, strict=False):
            b = Rectangle(
                width=sbw, height=sy(v) - sy0, stroke_width=0, fill_color=c, fill_opacity=1
            )
            s_bars.add(b.move_to([x, sy0, 0], aligned_edge=DOWN))
            s_names.add(txt(name, size=22, color=INK).next_to([x, sy0, 0], DOWN, buff=0.25))
            s_vals.add(txt(f"{v:.1f}", size=30, weight=SEMIBOLD).next_to(b, UP, buff=0.15))
        steps = VGroup(*[
            DashedLine([xs[i] + sbw / 2, sy(STAIRS[i][1]), 0], [xs[i + 1] - sbw / 2, sy(STAIRS[i][1]), 0],
                       dash_length=0.07, stroke_color=MUTED, stroke_width=1.5)
            for i in range(3)
        ])  # fmt: skip

        def reveal(i, rt=1.2):
            anims = [GrowFromEdge(s_bars[i], DOWN), FadeIn(s_names[i], shift=0.1 * UP)]
            if i:
                anims.append(Create(steps[i - 1]))
            self.play(*anims, run_time=rt, rate_func=SMOOTH)
            self.play(FadeIn(s_vals[i], shift=0.1 * UP), run_time=0.6, rate_func=SMOOTH)

        self.at(7)
        reveal(0)
        self.at(8)
        reveal(1)
        self.at(9)
        reveal(2)
        self.at(9, 3.6)
        reveal(3)

        # the middle step: adapting the encoder as well
        self.at(10)
        others = VGroup(
            s_bars[0], s_bars[3], s_vals[0], s_vals[3], s_names[0], s_names[3], steps[0], steps[2]
        )
        drop = Arrow([xs[2] - sbw / 2 - 0.35, sy(20.2), 0], [xs[2] - sbw / 2 - 0.35, sy(16.0), 0], buff=0,
                     stroke_width=4, stroke_color=ENC, tip_length=0.18,
                     max_tip_length_to_length_ratio=0.4)  # fmt: skip
        focus = RoundedRectangle(width=xs[2] - xs[1] + sbw + 0.8, height=sy(20.2) - sy0 + 1.6, corner_radius=0.2,
                                 stroke_color=ENC, stroke_width=2.5, fill_opacity=0)  # fmt: skip
        focus.move_to([(xs[1] + xs[2]) / 2, (sy(20.2) + 0.75 + sy0 - 0.75) / 2, 0])
        enc_n = txt("adapt the encoder too", size=20, color=INK2, t2c={"encoder": ENC}).next_to(
            focus, UP, buff=0.15
        )
        self.play(others.animate.set_opacity(0.25), Create(focus), run_time=1.2, rate_func=SMOOTH)
        self.play(GrowArrow(drop), FadeIn(enc_n, shift=0.1 * DOWN), run_time=0.9, rate_func=SMOOTH)
        self.at(11, 1.5)
        fifth = txt("another fifth of the errors removed", size=20, weight=SEMIBOLD)
        VGroup(enc_n.generate_target(), fifth).arrange(DOWN, buff=0.1).next_to(focus, UP, buff=0.15)
        self.play(
            MoveToTarget(enc_n), FadeIn(fifth, shift=0.1 * UP), run_time=1.0, rate_func=SMOOTH
        )

        # the question the rest of the talk answers
        self.at(12)
        stair = VGroup(s_title, s_base, s_bars, s_vals, s_names, steps, drop, focus, fifth, enc_n)
        self.play(FadeOut(stair, shift=0.15 * UP), run_time=1.0, rate_func=SMOOTH)
        q1 = txt("Where should a speech model", size=44, weight=SEMIBOLD)
        q2 = txt("learn a new language?", size=44, weight=SEMIBOLD, t2c={"language": ENC})
        q = VGroup(q1, q2).arrange(DOWN, buff=0.2).move_to([0, 0.3, 0])
        self.at(13, -0.4)
        self.play(LaggedStart(*[FadeIn(w, shift=0.2 * UP) for w in (q1, q2)], lag_ratio=0.35), run_time=1.6,
                  rate_func=SMOOTH)  # fmt: skip
        self.end()
