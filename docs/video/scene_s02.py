"""S02: the problem. Puno, the provided data, why Quechua is hard, sentence reuse, the test set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    AnimationGroup,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
)
from style import CARD, GRID, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt

# Peru's border (lon, lat): Natural Earth 1:50m Admin 0 - Countries, public domain
# (data/peru_naturalearth_50m.json, extracted from naturalearthdata.com).
PERU = [tuple(p) for p in json.loads((Path(__file__).parent / "data" / "peru_naturalearth_50m.json")
                                     .read_text())["lonlat"]]  # fmt: skip
PUNO = (-70.02, -15.84)  # Puno city


def to_xy(lon: float, lat: float, origin=(-75.0, -9.5), scale: float = 0.29) -> list[float]:
    return [(lon - origin[0]) * scale, (lat - origin[1]) * scale, 0]


def data_bar(hours: float, label: str, detail: str, color: str, hollow: bool = False) -> VGroup:
    """A horizontal bar whose length is proportional to hours of audio, with its label above."""
    w = hours * 0.135
    bar = Rectangle(
        width=w,
        height=0.42,
        stroke_width=2.5 if hollow else 0,
        stroke_color=color,
        fill_color=color,
        fill_opacity=0.18 if hollow else 1.0,
    )
    hrs = txt(f"{hours:g} h", size=26, color=INK, weight=SEMIBOLD).next_to(bar, RIGHT, buff=0.18)
    head = VGroup(
        txt(label, size=22, color=INK, weight=SEMIBOLD), txt(detail, size=20, color=MUTED)
    )
    head.arrange(RIGHT, buff=0.22, aligned_edge=DOWN)
    head.next_to(bar, UP, buff=0.14, aligned_edge=LEFT)
    return VGroup(bar, hrs, head)


class S02(TalkScene):
    sid = "S02"

    def construct(self) -> None:
        self.intro_chip()

        # --- Puno on the map, speakers ---------------------------------------------------------
        peru = Polygon(
            *[to_xy(*p) for p in PERU],
            color=REF,
            stroke_width=1.5,
            fill_color=CARD,
            fill_opacity=1.0,
        )
        puno = Dot(to_xy(*PUNO), radius=0.11, color=INK)
        VGroup(peru, puno).shift(3.6 * LEFT + 0.1 * DOWN)
        ring = Circle(radius=0.11, color=INK, stroke_width=3).move_to(puno)
        puno_lbl = txt("Puno", size=22, color=INK, weight=SEMIBOLD).next_to(puno, RIGHT, buff=0.18)
        self.play(FadeIn(peru, scale=0.97), run_time=1.2, rate_func=SMOOTH)
        self.play(FadeIn(puno, scale=0.3), FadeIn(puno_lbl, shift=0.1 * LEFT), run_time=0.7)
        self.play(ring.animate.scale(4).set_stroke(opacity=0), run_time=1.2, rate_func=SMOOTH)
        self.remove(ring)

        big = txt("≈474,000", size=72, color=INK, weight=SEMIBOLD)
        cap = txt("native Quechua speakers in the Puno region", size=28, color=INK2)
        speakers = (
            VGroup(big, cap).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([2.6, 0.6, 0])
        )
        sub = txt("very little speech technology", size=24, color=MUTED).next_to(
            speakers, DOWN, 0.5, LEFT
        )
        # the speaker count counts up, then the static text takes its place
        n = ValueTracker(0)
        counter = always_redraw(
            lambda: txt(
                f"≈{round(n.get_value(), -3):,.0f}", size=72, color=INK, weight=SEMIBOLD
            ).move_to(big, aligned_edge=LEFT)
        )
        self.add(counter)
        self.play(
            FadeIn(cap, shift=0.15 * UP),
            n.animate.set_value(474_000),
            run_time=1.8,
            rate_func=SMOOTH,
        )
        self.remove(counter)
        self.add(big)
        self.at(0, 4.0)
        self.play(FadeIn(sub, shift=0.1 * UP), run_time=0.7)

        # --- the provided data: three blocks, length = hours -------------------------------------
        self.at(1)
        self.play(FadeOut(VGroup(speakers, sub), shift=0.15 * UP), run_time=0.6, rate_func=SMOOTH)
        title = txt("Provided data (Common Voice)", size=30, color=INK, weight=SEMIBOLD)
        title.move_to([-0.6, 2.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(title, shift=0.1 * UP), run_time=0.7)
        read = data_bar(30.5, "Read speech", "77 speakers", INK2)
        gold = data_bar(5.2, "Spontaneous, transcribed", "", INK)
        silver = data_bar(37, "Spontaneous, untranscribed", "", REF, hollow=True)
        blocks = VGroup(read, gold, silver).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        blocks.next_to(title, DOWN, buff=0.55, aligned_edge=LEFT)

        def reveal(b: VGroup) -> None:
            self.play(
                GrowFromEdge(b[0], LEFT),
                FadeIn(b[2], shift=0.1 * UP),
                run_time=1.0,
                rate_func=SMOOTH,
            )
            self.play(FadeIn(b[1], shift=0.1 * LEFT), run_time=0.4)

        self.at(2)
        reveal(read)
        self.at(5)
        reveal(gold)
        self.at(5, 2.0)
        reveal(silver)

        # --- why Quechua is hard: suffixes --------------------------------------------------------
        self.at(6)
        self.play(
            FadeOut(VGroup(peru, puno, puno_lbl, title, blocks), shift=0.15 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        head = txt("Quechua is a hard case", size=40, color=INK, weight=SEMIBOLD).move_to(
            [0, 2.3, 0]
        )
        self.play(FadeIn(head, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)

        self.at(7)
        # one Text, sliced into morphemes by glyph, so the letters keep their common baseline
        full = txt("wasiykunapi", size=56, color=INK, weight=SEMIBOLD).move_to([0, 0.75, 0])
        parts = VGroup(full[0:4], full[4:5], full[5:9], full[9:11])
        word = parts.copy()
        self.play(FadeIn(word, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)
        trans = txt("“in my houses”", size=24, color=MUTED).next_to(word, DOWN, buff=0.3)
        self.play(FadeIn(trans), run_time=0.5)
        # split the word into its suffixes, with glosses
        spaced = parts.copy()
        for i in range(1, 4):
            spaced[i].next_to(spaced[i - 1], RIGHT, buff=0.42).set_y(parts[i].get_y())
        spaced.set_x(0)
        for i in range(1, 4):
            spaced[i].set_color(INK2)
        # glosses sliced from one Text too (shared baseline), each centred under its morpheme
        gt = txt("housemypluralinfrom", size=20, color=MUTED).next_to(spaced, DOWN, buff=0.3)
        gl = VGroup(gt[0:5], gt[5:7], gt[7:13], gt[13:15])
        for g, m in zip(gl, spaced, strict=True):
            g.set_x(m.get_x())
        self.play(Transform(word, spaced), FadeOut(trans), run_time=1.0, rate_func=SMOOTH)
        self.play(
            LaggedStart(*[FadeIn(g, shift=0.08 * DOWN) for g in gl], lag_ratio=0.2), run_time=0.9
        )
        # one wrong suffix: -pi "in" heard as -manta "from" -> the whole word is wrong
        self.at(7, 3.0)
        alt = txt("wasiykunamanta", size=56, color=INK2, weight=SEMIBOLD).move_to(
            full, aligned_edge=LEFT
        )
        wrong = alt[9:14].shift((word[3].get_left() - alt[9].get_left())[0] * RIGHT)
        wgl = gt[15:19].set_x(wrong.get_x())
        self.play(Transform(word[3], wrong), Transform(gl[3], wgl), run_time=0.8, rate_func=SMOOTH)
        strike = Line(word.get_left(), word.get_right(), color=INK, stroke_width=4).set_y(
            word[0].get_y()
        )
        verdict = txt("one wrong suffix  =  one wrong word", size=26, color=INK, weight=SEMIBOLD)
        verdict.next_to(gl, DOWN, buff=0.45)
        self.play(Create(strike), FadeIn(verdict, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        # --- stop consonants ---------------------------------------------------------------------
        self.at(8)
        self.play(
            FadeOut(VGroup(word, gl, strike, verdict), shift=0.15 * UP),
            run_time=0.7,
            rate_func=SMOOTH,
        )
        cols = ["plain", "aspirated", "ejective"]
        rows = [("t", "th", "t'"), ("k", "kh", "k'"), ("q", "qh", "q'")]
        x0, dx, y0, dy = -1.3, 2.3, 0.35, -1.0
        col_h = VGroup(
            *[
                txt(c, size=22, color=MUTED, weight=SEMIBOLD).move_to([x0 + j * dx, y0 + 0.95, 0])
                for j, c in enumerate(cols)
            ]
        )
        cells = VGroup(
            *[
                VGroup(
                    *[
                        txt(s, size=46, color=INK, weight=SEMIBOLD).move_to(
                            [x0 + j * dx, y0 + i * dy, 0]
                        )
                        for j, s in enumerate(r)
                    ]
                )
                for i, r in enumerate(rows)
            ]
        )
        row_h = VGroup(
            txt("dental", size=20, color=MUTED),
            txt("velar", size=20, color=MUTED),
            txt("uvular", size=20, color=MUTED),
        )
        for i, r in enumerate(row_h):
            r.move_to([x0 - 2.0, y0 + i * dy, 0])
        rules = VGroup(
            *[
                Line(
                    [x0 - 2.7, y0 + (i + 0.5) * dy, 0],
                    [x0 + 2 * dx + 0.9, y0 + (i + 0.5) * dy, 0],
                    color=GRID,
                    stroke_width=2,
                )
                for i in range(2)
            ]
        )
        plain = VGroup(cells[0][0], cells[1][0])
        self.play(FadeIn(plain, shift=0.1 * UP), FadeIn(col_h[0]), FadeIn(row_h[:2]), run_time=0.7)
        self.at(9)
        self.play(
            LaggedStart(
                *[
                    FadeIn(VGroup(col_h[j], cells[0][j], cells[1][j]), shift=0.1 * RIGHT)
                    for j in (1, 2)
                ],
                lag_ratio=0.5,
            ),
            Create(rules[0]),
            run_time=1.4,
        )
        self.at(9, 2.6)
        q_band = RoundedRectangle(
            corner_radius=0.12,
            width=2 * dx + 3.6,
            height=0.85,
            stroke_width=0,
            fill_color=INK2,
            fill_opacity=0.08,
        )
        q_band.move_to([x0 + dx - 0.9, y0 + 2 * dy, 0])
        self.play(
            FadeIn(q_band),
            Create(rules[1]),
            FadeIn(row_h[2]),
            LaggedStart(*[FadeIn(c, shift=0.1 * UP) for c in cells[2]], lag_ratio=0.25),
            run_time=1.2,
        )
        note = txt("contrasts Spanish and English lack", size=22, color=MUTED).move_to(
            [0, -2.55, 0]
        )
        self.play(FadeIn(note, shift=0.08 * UP), run_time=0.6)

        # --- sentence reuse: 22,727 clips collapse into 2,051 sentences ---------------------------
        # 24 x 11 clip dots; each column (one sentence's ~11 readings) merges into one sentence dot.
        # 264 : 24 keeps the real ratio, 22,727 / 2,051 = 11.1 readings per sentence.
        self.at(10)
        self.play(
            FadeOut(VGroup(head, col_h, cells, row_h, rules, q_band, note), shift=0.15 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        hours = txt("30.5 h of read speech", size=40, color=INK, weight=SEMIBOLD)
        hours.move_to([0, 2.4, 0])
        self.play(FadeIn(hours, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)
        n_cols, n_rows, step, gy = 24, 11, 0.25, 0.1
        y_of = [gy + (r - (n_rows - 1) / 2) * step for r in range(n_rows)]
        x_of = [(c - (n_cols - 1) / 2) * step for c in range(n_cols)]
        clips = VGroup(
            *[
                VGroup(*[Dot([x_of[c], y, 0], radius=0.07, color=REF) for y in y_of])
                for c in range(n_cols)
            ]
        )
        clips_lbl = txt("22,727 clips", size=30, color=INK2, weight=SEMIBOLD)
        clips_lbl.move_to([0, -1.75, 0])
        self.play(
            LaggedStart(*[FadeIn(col, scale=0.6) for col in clips], lag_ratio=0.06),
            FadeIn(clips_lbl, shift=0.1 * UP),
            run_time=1.6,
            rate_func=SMOOTH,
        )
        # "...cover only about two thousand distinct sentences"
        self.at(11, 2.0)
        sents = VGroup(*[Dot([x_of[c], gy, 0], radius=0.09, color=INK) for c in range(n_cols)])
        sent_lbl = txt("only 2,051 distinct sentences", size=36, color=INK, weight=SEMIBOLD)
        sent_lbl.move_to(clips_lbl)
        self.play(
            LaggedStart(
                *[ReplacementTransform(col, d) for col, d in zip(clips, sents, strict=True)],
                lag_ratio=0.04,
            ),
            AnimationGroup(
                FadeOut(clips_lbl, shift=0.2 * UP), FadeIn(sent_lbl, shift=0.2 * UP), lag_ratio=0.6
            ),
            run_time=1.8,
            rate_func=SMOOTH,
        )
        # "...so every sentence is read many times by many people": one sentence, its readings
        self.at(11, 5.6)
        k = 15
        hub = sents[k]
        readers = VGroup(
            *[
                Dot(
                    hub.get_center() + 1.1 * np.array([np.cos(a), np.sin(a), 0]), radius=0.07
                ).set_color(REF)
                for a in np.linspace(np.pi / 9, 8 * np.pi / 9, 11)
            ]
        )
        spokes = VGroup(
            *[Line(hub.get_center(), d.get_center(), color=GRID, stroke_width=2) for d in readers]
        )
        each = txt("each read ≈ 11 times, by many people", size=24, color=INK2)
        each.next_to(sent_lbl, DOWN, buff=0.3)
        others = VGroup(*[d for i, d in enumerate(sents) if i != k])
        self.play(others.animate.set_opacity(0.2), run_time=0.6, rate_func=SMOOTH)
        self.add(spokes, hub)
        self.play(
            LaggedStart(*[Create(sp) for sp in spokes], lag_ratio=0.08),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in readers], lag_ratio=0.08),
            FadeIn(each, shift=0.08 * UP),
            run_time=1.4,
            rate_func=SMOOTH,
        )

        # --- the test set is different ------------------------------------------------------------
        self.at(12)
        self.play(
            FadeOut(VGroup(hours, sents, spokes, readers, sent_lbl, each), shift=0.15 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        thead = VGroup(
            txt("Test set", size=40, color=INK, weight=SEMIBOLD),
            txt("659 clips", size=26, color=MUTED),
        )
        thead.arrange(RIGHT, buff=0.35, aligned_edge=DOWN).move_to([0, 2.25, 0])
        self.play(FadeIn(thead, shift=0.15 * UP), run_time=0.8, rate_func=SMOOTH)
        self.at(13)
        one = txt("1   its sentences are new", size=30, color=INK).move_to(
            [-3.6, 1.15, 0], aligned_edge=LEFT
        )
        self.play(FadeIn(one, shift=0.1 * UP), run_time=0.6)

        self.at(14)
        two = txt("2   its spontaneous speakers are older", size=30, color=INK)
        two.move_to([-3.6, 0.35, 0], aligned_edge=LEFT)
        self.play(FadeIn(two, shift=0.1 * UP), run_time=0.6)
        # an age axis, 20 to 80, with where each group sits
        ax_y, a0, a1 = -1.75, -3.6, 4.2

        def ax_x(age: float) -> float:
            return a0 + (age - 20) / 60 * (a1 - a0)

        axis = Line([a0, ax_y, 0], [a1, ax_y, 0], color=INK2, stroke_width=2)
        ticks = VGroup(
            *[
                VGroup(
                    Line([ax_x(a), ax_y, 0], [ax_x(a), ax_y - 0.1, 0], color=INK2, stroke_width=2),
                    txt(str(a) if a < 80 else "80+", size=18, color=MUTED).move_to(
                        [ax_x(a), ax_y - 0.35, 0]
                    ),
                )
                for a in (20, 40, 60, 80)
            ]
        )
        age = txt("speaker age", size=18, color=MUTED).next_to(axis, RIGHT, buff=0.25)
        self.play(Create(axis), FadeIn(ticks), FadeIn(age), run_time=0.8, rate_func=SMOOTH)
        train = Rectangle(
            width=ax_x(40) - ax_x(18), height=0.36, stroke_width=0, fill_color=REF, fill_opacity=1
        )
        train.move_to([(ax_x(18) + ax_x(40)) / 2, ax_y + 0.95, 0])
        test = Rectangle(
            width=ax_x(80) - ax_x(60), height=0.36, stroke_width=0, fill_color=INK, fill_opacity=1
        )
        test.move_to([(ax_x(60) + ax_x(80)) / 2, ax_y + 0.45, 0])
        train_l = txt("training transcripts: 1,069 of 1,074 clips under 40", size=20, color=INK2)
        train_l.next_to(train, RIGHT, buff=0.2)
        test_l = txt("test speakers: all 60+", size=20, color=INK, weight=SEMIBOLD).next_to(
            test, LEFT, buff=0.2
        )
        self.play(
            GrowFromEdge(test, LEFT),
            FadeIn(test_l, shift=0.1 * RIGHT),
            run_time=0.9,
            rate_func=SMOOTH,
        )
        self.at(14, 3.2)
        self.play(
            GrowFromEdge(train, LEFT),
            FadeIn(train_l, shift=0.1 * LEFT),
            run_time=0.9,
            rate_func=SMOOTH,
        )
        self.end()
