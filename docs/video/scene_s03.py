"""S03: the sentence-reuse trap. A prompt-trained 4-gram wins on seen sentences, loses on new."""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    NORMAL,
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
    Text,
    TransformMatchingShapes,
    VGroup,
)
from style import CARD, GRID, INK, INK2, LM4, MUTED, REF, SMOOTH, TalkScene, txt

# WER (%) on the dev set, seen vs held-out sentences (reports/fig_data/sentence_reuse.csv)
LM_SEEN, LM_NEW = 6.0, 18.7
GREEDY_SEEN, GREEDY_NEW = 14.1, 14.6
OURS_SEEN, OURS_NEW = 10.0, 9.7

# slope-chart geometry
X_SEEN, X_NEW = -2.9, 1.1
Y0, YS = -2.55, 0.225  # y of WER 0, units per WER point


def py(wer: float) -> float:
    return Y0 + wer * YS


def sentence_card(label: str) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.08,
        width=2.0,
        height=0.46,
        stroke_width=0,
        fill_color=CARD,
        fill_opacity=1.0,
    )
    return VGroup(box, txt(label, size=18, color=INK2).move_to(box))


class S03(TalkScene):
    sid = "S03"

    def construct(self) -> None:
        self.intro_chip()

        # --- the official splits: speakers apart, sentences shared --------------------------------
        xt, xs = -1.6, 2.4  # train and test columns
        heads = VGroup(
            txt("training", size=24, color=INK2, weight=SEMIBOLD).move_to([xt, 2.2, 0]),
            txt("test", size=24, color=INK2, weight=SEMIBOLD).move_to([xs, 2.2, 0]),
        )
        divider = DashedLine(
            [(xt + xs) / 2, 2.5, 0], [(xt + xs) / 2, -2.4, 0], color=GRID, stroke_width=2
        )
        row_l = VGroup(
            txt("speakers", size=22, color=MUTED).move_to([-5.3, 1.1, 0], aligned_edge=LEFT),
            txt("sentences", size=22, color=MUTED).move_to([-5.3, -0.75, 0], aligned_edge=LEFT),
        )
        spk_train = VGroup(*[Dot(radius=0.14, color=REF) for _ in range(5)]).arrange(
            RIGHT, buff=0.22
        )
        spk_test = VGroup(*[Dot(radius=0.14, color=INK) for _ in range(5)]).arrange(
            RIGHT, buff=0.22
        )
        spk_train.move_to([xt, 1.1, 0])
        spk_test.move_to([xs, 1.1, 0])
        names = ["sentence A", "sentence B", "sentence C"]
        s_train = (
            VGroup(*[sentence_card(n) for n in names])
            .arrange(DOWN, buff=0.16)
            .move_to([xt, -0.75, 0])
        )
        s_test = (
            VGroup(*[sentence_card(n) for n in names])
            .arrange(DOWN, buff=0.16)
            .move_to([xs, -0.75, 0])
        )
        links = VGroup(
            *[
                Line(a.get_right(), b.get_left(), color=INK2, stroke_width=2)
                for a, b in zip(s_train, s_test, strict=True)
            ]
        )
        apart = txt("kept apart", size=24, color=INK, weight=SEMIBOLD).move_to(
            [4.6, 1.1, 0], aligned_edge=LEFT
        )
        shared = txt("all shared", size=24, color=INK, weight=SEMIBOLD).move_to(
            [4.6, -0.75, 0], aligned_edge=LEFT
        )
        self.play(FadeIn(heads, shift=0.1 * DOWN), Create(divider), run_time=0.9, rate_func=SMOOTH)
        self.play(
            FadeIn(row_l[0]),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in [*spk_train, *spk_test]], lag_ratio=0.06),
            run_time=1.0,
        )
        self.play(FadeIn(apart, shift=0.1 * LEFT), run_time=0.5)
        self.at(0, 3.4)
        self.play(FadeIn(row_l[1]), FadeIn(s_train, shift=0.1 * UP), run_time=0.7)
        self.play(
            LaggedStart(*[Create(ln) for ln in links], lag_ratio=0.2),
            FadeIn(s_test, shift=0.1 * LEFT),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        self.play(FadeIn(shared, shift=0.1 * LEFT), run_time=0.5)
        split = VGroup(
            heads, divider, row_l, spk_train, spk_test, s_train, s_test, links, apart, shared
        )

        # --- word error rate ----------------------------------------------------------------------
        self.at(2)
        self.play(FadeOut(split, shift=0.15 * UP), run_time=0.7, rate_func=SMOOTH)
        wer = txt("Word error rate (WER)", size=42, color=INK, weight=SEMIBOLD).move_to([0, 1.0, 0])
        frac = txt("share of words the system gets wrong", size=30, color=INK2).next_to(
            wer, DOWN, buff=0.45
        )
        lower = txt("lower is better", size=26, color=MUTED).next_to(frac, DOWN, buff=0.45)
        self.play(FadeIn(wer, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)
        self.at(3)
        self.play(FadeIn(frac, shift=0.1 * UP), run_time=0.7)
        self.play(FadeIn(lower, shift=0.1 * DOWN), run_time=0.6)

        # --- the slope chart: seen vs new sentences -----------------------------------------------
        self.at(4)
        self.play(
            FadeOut(VGroup(wer, frac, lower), shift=0.15 * UP), run_time=0.7, rate_func=SMOOTH
        )
        grid = VGroup(
            *[
                Line([-4.6, py(w), 0], [X_NEW + 0.5, py(w), 0], color=GRID, stroke_width=1.5)
                for w in (0, 5, 10, 15, 20)
            ]
        )
        ylab = VGroup(
            *[
                txt(str(w), size=18, color=MUTED).move_to([-4.95, py(w), 0])
                for w in (0, 5, 10, 15, 20)
            ]
        )
        ytitle = txt("WER (%), dev set", size=20, color=MUTED).move_to(
            [-5.3, py(20) + 0.5, 0], aligned_edge=LEFT
        )
        xlab = VGroup(
            txt("seen sentences", size=24, color=INK2, weight=SEMIBOLD).move_to(
                [X_SEEN, Y0 - 0.45, 0]
            ),
            txt("new sentences", size=24, color=INK2, weight=SEMIBOLD).move_to(
                [X_NEW, Y0 - 0.45, 0]
            ),
        )
        system = txt(
            "speech model  +  word 4-gram on the training prompts",
            size=24,
            color=INK2,
            t2c={"4-gram": LM4},
        ).move_to([0, 3.0, 0])
        self.play(FadeIn(system, shift=0.1 * DOWN), run_time=0.8, rate_func=SMOOTH)
        self.play(
            FadeIn(grid), FadeIn(ylab), FadeIn(ytitle), FadeIn(xlab, shift=0.1 * UP), run_time=1.0
        )

        def point(x: float, wer: float, color: str) -> Dot:
            return Dot([x, py(wer), 0], radius=0.1, color=color)

        def value(d: Dot, wer: float, side=LEFT) -> Text:
            return txt(f"{wer:.1f}", size=22, color=INK, weight=SEMIBOLD).next_to(
                d, side, buff=0.16
            )

        def tag(d: Dot, label: str) -> Text:
            return txt(label, size=20, color=INK2).next_to(d, RIGHT, buff=0.2)

        # on seen sentences: three systems, the 4-gram lowest
        gr_a, us_a, lm_a = (
            point(X_SEEN, GREEDY_SEEN, REF),
            point(X_SEEN, OURS_SEEN, INK),
            point(X_SEEN, LM_SEEN, LM4),
        )
        gr_av, us_av, lm_av = value(gr_a, GREEDY_SEEN), value(us_a, OURS_SEEN), value(lm_a, LM_SEEN)
        gr_t, us_t, lm_t = tag(gr_a, "no LM"), tag(us_a, "ours"), tag(lm_a, "+ 4-gram")
        self.at(5)
        self.play(
            LaggedStart(
                *[FadeIn(VGroup(d, v, t), shift=0.1 * UP) for d, v, t in
                  ((gr_a, gr_av, gr_t), (us_a, us_av, us_t))],
                lag_ratio=0.3,
            ),
            run_time=0.9,
            rate_func=SMOOTH,
        )  # fmt: skip
        self.play(
            FadeIn(VGroup(lm_a, lm_av, lm_t), shift=0.2 * DOWN), run_time=0.8, rate_func=SMOOTH
        )
        # "six percent": the others step back
        self.at(6)
        rest_a = VGroup(gr_a, gr_av, gr_t, us_a, us_av, us_t)
        self.play(rest_a.animate.set_opacity(0.25), run_time=0.7, rate_func=SMOOTH)

        # "on new sentences it jumps...": the flat lines draw first, then the 4-gram climbs across
        self.at(7)
        gr_b, us_b, lm_b = (
            point(X_NEW, GREEDY_NEW, REF),
            point(X_NEW, OURS_NEW, INK),
            point(X_NEW, LM_NEW, LM4),
        )
        gr_line = Line(gr_a.get_center(), gr_b.get_center(), color=REF, stroke_width=5)
        us_line = Line(us_a.get_center(), us_b.get_center(), color=INK, stroke_width=5)
        lm_line = Line(lm_a.get_center(), lm_b.get_center(), color=LM4, stroke_width=5)
        gr_bv, us_bv, lm_bv = (
            value(gr_b, GREEDY_NEW, RIGHT),
            value(us_b, OURS_NEW, RIGHT),
            value(lm_b, LM_NEW, RIGHT),
        )
        gr_name = txt("no language model", size=20, color=INK2).next_to(gr_bv, RIGHT, buff=0.25)
        us_name = txt("our system", size=20, color=INK, weight=SEMIBOLD).next_to(us_bv, RIGHT, 0.25)
        lm_name = txt("4-gram on the prompts", size=20, color=INK, weight=SEMIBOLD)
        lm_name.next_to(lm_bv, RIGHT, buff=0.25)

        def run(line: Line, start: Dot, end: Dot, run_time: float, *extra) -> None:
            """Draw a line from its seen point while a head dot rides its tip to the new point."""
            head = start.copy()
            self.add(line, start, head)
            self.play(
                Create(line), head.animate.move_to(end), *extra, run_time=run_time, rate_func=SMOOTH
            )
            self.remove(head)
            self.add(end)

        self.play(
            rest_a.animate.set_opacity(1),
            FadeOut(VGroup(gr_t, us_t, lm_t)),
            run_time=0.5,
            rate_func=SMOOTH,
        )
        heads = VGroup(gr_a.copy(), us_a.copy())
        self.add(gr_line, us_line, gr_a, us_a, heads)
        self.play(
            Create(gr_line),
            Create(us_line),
            heads[0].animate.move_to(gr_b),
            heads[1].animate.move_to(us_b),
            run_time=1.1,
            rate_func=SMOOTH,
        )
        self.remove(heads)
        self.add(gr_b, us_b)
        self.play(FadeIn(VGroup(gr_bv, us_bv, gr_name, us_name), shift=0.1 * LEFT), run_time=0.5)
        run(lm_line, lm_a, lm_b, 1.7)
        self.play(FadeIn(VGroup(lm_bv, lm_name), shift=0.1 * LEFT), run_time=0.6)
        # "...worse than the same model with no language model at all"
        self.at(7, 4.6)
        ours = VGroup(us_a, us_b, us_line, us_av, us_bv, us_name)
        self.play(ours.animate.set_opacity(0.25), run_time=0.7, rate_func=SMOOTH)

        # "hasn't learned Quechua. It has memorized the prompts." / ours generalizes
        self.at(8)
        us_name2 = txt("our system: generalizes", size=20, color=INK, weight=SEMIBOLD)
        us_name2.move_to(us_name, aligned_edge=LEFT)
        self.play(ours.animate.set_opacity(1), run_time=0.6, rate_func=SMOOTH)
        self.play(TransformMatchingShapes(us_name, us_name2), run_time=0.8, rate_func=SMOOTH)
        us_name = us_name2
        self.at(9)
        lm_name2 = txt("4-gram: memorizes the prompts", size=20, color=INK, weight=SEMIBOLD)
        lm_name2.move_to(lm_name, aligned_edge=LEFT)
        self.play(TransformMatchingShapes(lm_name, lm_name2), run_time=0.8, rate_func=SMOOTH)
        lm_name = lm_name2

        # the LM's effect flips sign: -8.1 on seen sentences, +4.1 on new ones
        self.at(10)
        off = 0.42 * LEFT
        up_seen = Arrow(
            gr_a.get_center() + off,
            lm_a.get_center() + off,
            buff=0.05,
            color=INK2,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.12,
        )
        up_new = Arrow(
            gr_b.get_center() + off,
            lm_b.get_center() + off,
            buff=0.05,
            color=INK2,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.3,
        )
        d_seen = txt("LM −8.1", size=18, color=INK2).next_to(up_seen, LEFT, buff=0.1)
        d_new = (
            txt("LM +4.1", size=18, color=INK2)
            .next_to(up_new.get_top(), LEFT, buff=0.12)
            .shift(0.25 * UP)
        )
        # the seen-side values make room for the arrow; our system steps back
        self.play(
            FadeOut(VGroup(gr_av, us_av, lm_av)),
            VGroup(us_a, us_b, us_line, us_bv, us_name).animate.set_opacity(0.25),
            GrowArrow(up_seen),
            FadeIn(d_seen),
            run_time=0.9,
        )
        self.play(GrowArrow(up_new), FadeIn(d_new), run_time=0.7)
        box = RoundedRectangle(
            corner_radius=0.12,
            width=4.6,
            height=1.25,
            stroke_width=0,
            fill_color=CARD,
            fill_opacity=1.0,
        ).move_to([4.0, -1.75, 0])
        rev = txt("reversal: −12.1 points", size=28, color=INK, weight=SEMIBOLD)
        ci = txt("95% interval [−18.0, −7.1]", size=20, color=INK2)
        VGroup(rev, ci).arrange(DOWN, buff=0.14).move_to(box)
        self.at(10, 2.5)
        self.play(
            FadeIn(box, scale=0.96), FadeIn(rev, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH
        )
        self.play(FadeIn(ci, shift=0.08 * UP), run_time=0.6)
        chart = VGroup(
            system,
            grid,
            ylab,
            ytitle,
            xlab,
            lm_a,
            lm_b,
            lm_line,
            lm_bv,
            lm_name,
            gr_a,
            gr_b,
            gr_line,
            gr_bv,
            gr_name,
            us_a,
            us_b,
            us_line,
            us_bv,
            us_name,
            up_seen,
            up_new,
            d_seen,
            d_new,
            box,
            rev,
            ci,
        )

        # --- the real test reads new sentences ----------------------------------------------------
        self.at(11)
        self.play(FadeOut(chart, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)
        title = txt("Distance to the nearest training prompt", size=34, color=INK, weight=SEMIBOLD)
        sub = txt("median normalized edit distance", size=22, color=MUTED)
        VGroup(title, sub).arrange(DOWN, buff=0.16).move_to([0, 2.15, 0])
        self.play(FadeIn(title, shift=0.15 * UP), FadeIn(sub), run_time=0.8, rate_func=SMOOTH)
        rows = [
            ("seen sentences", 0.028, REF),
            ("held-out sentences", 0.489, INK2),
            ("official test", 0.495, INK),
        ]
        bx, scale = -1.4, 9.0
        bars = VGroup()
        for i, (name, d, color) in enumerate(rows):
            y = 0.7 - i * 0.85
            lab = txt(
                name, size=24, color=INK if i == 2 else INK2, weight=SEMIBOLD if i == 2 else NORMAL
            )
            lab.move_to([bx - 0.3, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(
                width=max(d * scale, 0.04),
                height=0.48,
                stroke_width=0,
                fill_color=color,
                fill_opacity=1,
            )
            bar.move_to([bx, y, 0], aligned_edge=LEFT)
            val = txt(f"{d:.3f}", size=24, color=INK, weight=SEMIBOLD).next_to(
                bar, RIGHT, buff=0.18
            )
            bars.add(VGroup(lab, bar, val))

        def show(row: VGroup) -> None:
            self.play(
                FadeIn(row[0], shift=0.1 * RIGHT),
                GrowFromEdge(row[1], LEFT),
                run_time=0.8,
                rate_func=SMOOTH,
            )
            self.play(FadeIn(row[2], shift=0.1 * LEFT), run_time=0.4)

        show(bars[0])
        self.at(12)
        show(bars[1])
        self.at(12, 2.2)
        show(bars[2])
        self.at(13)
        verdict = txt(
            "the official splits would have picked the wrong system",
            size=28,
            color=INK,
            weight=SEMIBOLD,
        ).move_to([0, -2.1, 0])
        self.play(FadeIn(verdict, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.end()
