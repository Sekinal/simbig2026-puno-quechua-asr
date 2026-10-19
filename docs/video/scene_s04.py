"""S04: how we evaluate: held-out views, spontaneous speakers, the speaker x sentence bootstrap."""

from __future__ import annotations

import random

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    RoundedRectangle,
    Square,
    VGroup,
)
from style import CARD, INK, INK2, MUTED, SMOOTH, TalkScene, txt

ROWS, COLS = 6, 12  # bootstrap matrix: speakers x sentences


def eval_card(title: str, number: str, detail: str) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.14, width=3.9, height=2.3, stroke_width=0, fill_color=CARD, fill_opacity=1.0
    )
    t = txt(title, size=24, color=INK, weight=SEMIBOLD)
    n = txt(number, size=54, color=INK, weight=SEMIBOLD)
    d = txt(detail, size=19, color=INK2, line_spacing=0.8)
    VGroup(t, n, d).arrange(DOWN, buff=0.2).move_to(box)
    return VGroup(box, t, n, d)


def pill(label: str) -> VGroup:
    t = txt(label, size=22, color=INK, weight=SEMIBOLD)
    box = RoundedRectangle(
        corner_radius=0.25,
        width=t.width + 0.6,
        height=0.56,
        stroke_width=0,
        fill_color=CARD,
        fill_opacity=1.0,
    )
    return VGroup(box, t.move_to(box))


class S04(TalkScene):
    sid = "S04"

    def construct(self) -> None:
        self.intro_chip()

        head = txt("Our own evaluation", size=40, color=INK, weight=SEMIBOLD).move_to([0, 2.6, 0])
        self.play(FadeIn(head, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)

        # --- three held-out views -----------------------------------------------------------------
        cards = (
            VGroup(
                eval_card(
                    "Held-out sentences",
                    "400",
                    "sentences, never trained on\n(neither audio nor text)",
                ),
                eval_card(
                    "Held-out speakers",
                    "21",
                    "speakers in the internal test,\nnever seen in training",
                ),
                eval_card("Out of domain", "109", "radio drama clips,\nnever tuned on"),
            )
            .arrange(RIGHT, buff=0.35)
            .move_to([0, 0.75, 0])
        )
        for i in range(3):
            self.at(i + 1)
            self.play(FadeIn(cards[i], shift=0.3 * UP), run_time=0.9, rate_func=SMOOTH)

        # --- spontaneous speech: 4 + 6 speakers ---------------------------------------------------
        self.at(4)
        spon = txt("Spontaneous speech", size=24, color=INK, weight=SEMIBOLD).move_to(
            [-6.0, -1.25, 0], LEFT
        )
        dev = VGroup(*[Dot(radius=0.15, color=INK2) for _ in range(4)]).arrange(RIGHT, buff=0.16)
        test = VGroup(*[Dot(radius=0.15, color=INK) for _ in range(6)]).arrange(RIGHT, buff=0.16)
        dev_l = txt("4 dev speakers", size=20, color=MUTED)
        test_l = txt("6 test speakers", size=20, color=MUTED)
        dev_g = VGroup(dev, dev_l).arrange(DOWN, buff=0.16)
        test_g = VGroup(test, test_l).arrange(DOWN, buff=0.16)
        VGroup(dev_g, test_g).arrange(RIGHT, buff=0.7, aligned_edge=UP).next_to(
            spon, RIGHT, buff=0.7
        )
        VGroup(dev_g, test_g).align_to(spon, UP).shift(0.05 * UP)
        VGroup(spon, dev_g, test_g).set_x(0)
        self.play(FadeIn(spon, shift=0.1 * RIGHT), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in [*dev, *test]], lag_ratio=0.08),
            FadeIn(dev_l),
            FadeIn(test_l),
            run_time=1.3,
        )
        self.at(5)
        few = VGroup(
            txt("too few for strong conclusions:", size=22, color=INK2),
            txt(
                "descriptive, with an exact sign test over speakers",
                size=22,
                color=INK,
                weight=SEMIBOLD,
            ),
        )
        few.arrange(RIGHT, buff=0.25).next_to(VGroup(spon, dev_g, test_g), DOWN, buff=0.45).set_x(0)
        self.play(FadeIn(few[0], shift=0.1 * LEFT), run_time=0.6)
        self.at(5, 3.0)
        self.play(FadeIn(few[1], shift=0.1 * LEFT), run_time=0.6)

        # --- every comparison: a 95% interval from a paired bootstrap -----------------------------
        self.at(6)
        self.play(
            FadeOut(VGroup(cards, spon, dev_g, test_g, few), shift=0.15 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        pills = VGroup(pill("95% intervals"), pill("paired bootstrap"), pill("Holm correction"))
        pills.arrange(RIGHT, buff=0.3).move_to([0, 1.55, 0])
        self.play(
            LaggedStart(*[FadeIn(p, shift=0.15 * UP) for p in pills[:2]], lag_ratio=0.4),
            run_time=1.0,
        )
        self.at(6, 2.6)
        self.play(FadeIn(pills[2], shift=0.15 * UP), run_time=0.6)

        # the matrix of clips: one cell per (speaker, sentence)
        self.at(7)
        cells = VGroup(
            *[
                VGroup(
                    *[
                        Square(0.3, stroke_width=0, fill_color=CARD, fill_opacity=1.0)
                        for _ in range(COLS)
                    ]
                ).arrange(RIGHT, buff=0.07)
                for _ in range(ROWS)
            ]
        ).arrange(DOWN, buff=0.07)
        cells.move_to([0.4, -0.95, 0])
        rows_l = txt("speakers", size=20, color=MUTED).next_to(cells, LEFT, buff=0.75)
        cols_l = txt("sentences", size=20, color=MUTED).next_to(cells, UP, buff=0.75)
        cap = txt("resample speakers and sentences together", size=24, color=INK, weight=SEMIBOLD)
        cap.next_to(cells, DOWN, buff=0.4)
        self.play(
            FadeIn(cells, scale=0.97),
            FadeIn(rows_l),
            FadeIn(cols_l),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        self.play(FadeIn(cap, shift=0.1 * UP), run_time=0.6)

        # bootstrap draws: whole rows (speakers) and whole columns (sentences), with replacement.
        # Small dots beside a row / above a column count its draws; a cell's shade is
        # (row draws) x (column draws), i.e. how many times that clip enters the resample.
        def marks(counts: list[int], along_rows: bool) -> VGroup:
            out = VGroup()
            for i, k in enumerate(counts):
                if not k:
                    continue
                pips = VGroup(*[Dot(radius=0.045, color=INK) for _ in range(k)])
                if along_rows:
                    pips.arrange(LEFT, buff=0.06).next_to(cells[i][0], LEFT, buff=0.15)
                else:
                    pips.arrange(UP, buff=0.06).next_to(cells[0][i], UP, buff=0.15)
                out.add(pips)
            return out

        def shade(rc: list[int], cc: list[int]) -> list:
            anims = []
            for r in range(ROWS):
                for c in range(COLS):
                    k = rc[r] * cc[c]
                    color, op = (CARD, 1.0) if k == 0 else (INK2, min(1.0, 0.35 + 0.3 * k))
                    anims.append(cells[r][c].animate.set_fill(color, opacity=op))
            return anims

        rng = random.Random(4)
        r_marks, c_marks = VGroup(), VGroup()
        for i in range(3):
            r_draw = rng.choices(range(ROWS), k=ROWS)
            c_draw = rng.choices(range(COLS), k=COLS)
            rc = [r_draw.count(r) for r in range(ROWS)]
            cc = [c_draw.count(c) for c in range(COLS)]
            new_r, new_c = marks(rc, along_rows=True), marks(cc, along_rows=False)
            if i == 0:
                # first draw in two steps, so the unit is visible: speakers, then sentences
                self.play(
                    FadeIn(new_r, shift=0.1 * RIGHT),
                    *shade(rc, [1] * COLS),
                    run_time=0.8,
                    rate_func=SMOOTH,
                )
                self.wait(0.5)
                self.play(
                    FadeIn(new_c, shift=0.1 * DOWN), *shade(rc, cc), run_time=0.8, rate_func=SMOOTH
                )
                self.wait(0.9)
            else:
                self.play(
                    FadeOut(r_marks),
                    FadeOut(c_marks),
                    FadeIn(new_r),
                    FadeIn(new_c),
                    *shade(rc, cc),
                    run_time=0.7,
                    rate_func=SMOOTH,
                )
                self.wait(0.7)
            r_marks, c_marks = new_r, new_c
        why = txt(
            "clips from the same speaker or sentence are not independent", size=20, color=MUTED
        )
        why.next_to(cap, DOWN, buff=0.16)
        self.play(FadeIn(why, shift=0.08 * UP), run_time=0.6)
        self.end()
