"""S13: verification is the bottleneck. Better candidates exist; RL does not help; ranking accuracy is the limit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    UR,
    Arrow,
    Circle,
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
    VGroup,
    VMobject,
)
from style import CARD, INK, INK2, MUTED, REF, SMOOTH, SYN, TalkScene, txt

REPORTS = Path(__file__).resolve().parents[2] / "reports"

# A better transcript than the top-1 among 8 candidates, % of 150 held-out-sentence clips (paper §6).
FOUND = (("beam search", 6.0, REF), ("sampling", 14.0, MUTED), ("perturbed audio", 16.7, INK))
# Held-out sentences: best candidate in the pool vs our pick (verifier_curve.csv / verifier_ours_ci.csv).
ORACLE, OURS = 8.6, 16.9
RL = ("GSPO", "MWER", "Dr GRPO")
SET = "dev_unseen"


def curve(family: str) -> list[tuple[float, float]]:
    rows = [
        r
        for r in csv.DictReader(open(REPORTS / "verifier_curve.csv"))
        if r["set"] == SET and r["family"] == family
    ]
    rows.sort(key=lambda r: float(r["x"]), reverse=family == "noisy oracle")
    return [(float(r["pair_acc"]), float(r["wer"])) for r in rows]


def ours(name: str) -> dict:
    rows = {r["set"]: r for r in csv.DictReader(open(REPORTS / name))}
    return {k: float(v) for k, v in rows[SET].items() if k != "set"}


class S13(TalkScene):
    sid = "S13"

    def construct(self) -> None:
        self.intro_chip()

        # ---- A: better candidates exist ---------------------------------------------------------
        title = txt("A better transcript is among 8 candidates", size=30, weight=SEMIBOLD).move_to(
            [0, 2.35, 0]
        )
        sub = txt("share of clips, new sentences (%)", size=20, color=MUTED).next_to(
            title, DOWN, buff=0.16
        )
        bx0, bscale = -2.2, 0.33
        rows = VGroup()
        for (name, v, c), y in zip(FOUND, (0.85, -0.05, -0.95), strict=False):
            lab = txt(name, size=24, color=INK).move_to([bx0 - 0.3, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(
                width=v * bscale, height=0.52, stroke_width=0, fill_color=c, fill_opacity=0.9
            )
            bar.move_to([bx0, y, 0], aligned_edge=LEFT)
            val = txt(f"{v:.1f}%", size=24, color=INK).next_to(bar, RIGHT, buff=0.15)
            rows.add(VGroup(lab, bar, val))
        axis = Line([bx0, 1.35, 0], [bx0, -1.45, 0], stroke_color=INK2, stroke_width=1.5)
        easy = txt("Finding them is easy", size=28, t2c={"easy": INK}, color=INK2).move_to(
            [0, -2.4, 0]
        )

        self.at(0)
        self.play(
            FadeIn(title, shift=0.1 * UP),
            FadeIn(sub, shift=0.1 * UP),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        self.at(1)
        self.play(Create(axis), LaggedStart(*[FadeIn(r[0], shift=0.1 * RIGHT) for r in rows], lag_ratio=0.2),
                  run_time=1.1, rate_func=SMOOTH)  # fmt: skip
        self.at(2)
        self.play(GrowFromEdge(rows[0][1], LEFT), run_time=0.9, rate_func=SMOOTH)
        self.play(FadeIn(rows[0][2], shift=0.08 * RIGHT), run_time=0.5)
        self.at(2, 5.2)
        self.play(LaggedStart(GrowFromEdge(rows[1][1], LEFT), GrowFromEdge(rows[2][1], LEFT), lag_ratio=0.4),
                  run_time=1.6, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(rows[1][2], shift=0.08 * RIGHT), FadeIn(rows[2][2], shift=0.08 * RIGHT),
                  FadeIn(easy, shift=0.1 * UP), run_time=0.7)  # fmt: skip

        # ---- B: the best candidate in the pool halves the errors --------------------------------
        title2 = txt("If we could pick the best one", size=30, weight=SEMIBOLD).move_to(title)
        sub2 = txt("WER on new sentences (%)", size=20, color=MUTED).next_to(
            title2, DOWN, buff=0.16
        )
        rows2 = VGroup()
        for (name, v, c), y in zip(
            (("our pick", OURS, INK), ("best in our pool", ORACLE, SYN)), (0.6, -0.5), strict=False
        ):
            lab = txt(name, size=24, color=INK).move_to([bx0 - 0.3, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(
                width=v * bscale, height=0.6, stroke_width=0, fill_color=c, fill_opacity=0.9
            )
            bar.move_to([bx0, y, 0], aligned_edge=LEFT)
            val = txt(f"{v:.1f}", size=26, color=INK).next_to(bar, RIGHT, buff=0.15)
            rows2.add(VGroup(lab, bar, val))
        half_x = bx0 + OURS / 2 * bscale
        half = DashedLine(
            [half_x, 1.0, 0],
            [half_x, -0.95, 0],
            dash_length=0.07,
            stroke_color=INK2,
            stroke_width=2,
        )
        half_t = txt("half", size=22, color=INK2).next_to(half, DOWN, buff=0.12)
        self.at(3)
        self.play(
            FadeOut(rows, shift=0.15 * UP), FadeOut(easy, shift=0.15 * UP),
            FadeOut(title, shift=0.1 * UP), FadeOut(sub, shift=0.1 * UP),
            run_time=0.8, rate_func=SMOOTH,
        )  # fmt: skip
        self.play(FadeIn(title2, shift=0.1 * UP), FadeIn(sub2, shift=0.1 * UP),
                  LaggedStart(*[FadeIn(r[0], shift=0.1 * RIGHT) for r in rows2], lag_ratio=0.2),
                  run_time=0.9, rate_func=SMOOTH)  # fmt: skip
        self.play(GrowFromEdge(rows2[0][1], LEFT), run_time=0.9, rate_func=SMOOTH)
        self.play(FadeIn(rows2[0][2], shift=0.08 * RIGHT), run_time=0.4)
        self.at(4)
        self.play(GrowFromEdge(rows2[1][1], LEFT), run_time=0.9, rate_func=SMOOTH)
        self.play(
            FadeIn(rows2[1][2], shift=0.08 * RIGHT),
            Create(half),
            FadeIn(half_t),
            run_time=0.8,
            rate_func=SMOOTH,
        )

        # ---- C: reinforcement learning does not help --------------------------------------------
        title3 = txt("Reinforcement learning", size=30, weight=SEMIBOLD).move_to(title)
        chips = VGroup()
        for name in RL:
            box = RoundedRectangle(corner_radius=0.14, width=2.3, height=0.85, stroke_color=INK2, stroke_width=1.5,
                                   fill_color=CARD, fill_opacity=1)  # fmt: skip
            chips.add(VGroup(box, txt(name, size=28, weight=SEMIBOLD).move_to(box)))
        chips.arrange(RIGHT, buff=0.6).move_to([0, 1.2, 0])
        strikes = VGroup(
            *[Line(c.get_corner(DOWN + LEFT) + [0.2, 0.15, 0], c.get_corner(UP + RIGHT) + [-0.2, -0.15, 0],
                   stroke_color=INK2, stroke_width=3) for c in chips]
        )  # fmt: skip
        never = txt("never beat supervised training on new sentences", size=24, color=INK2).move_to(
            [0, 0.2, 0]
        )
        fine = txt("+0.3 to +1.6 WER points, not significant", size=18, color=MUTED).next_to(
            never, DOWN, buff=0.14
        )
        self.at(5)
        self.play(
            FadeOut(VGroup(title2, sub2, rows2, half, half_t, axis), shift=0.15 * UP),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        self.play(FadeIn(title3, shift=0.1 * UP), LaggedStart(*[FadeIn(c, shift=0.15 * UP) for c in chips],
                  lag_ratio=0.2), run_time=1.2, rate_func=SMOOTH)  # fmt: skip
        self.at(6, 0.4)
        self.play(LaggedStart(*[Create(s) for s in strikes], lag_ratio=0.3), chips.animate.set_opacity(0.45),
                  run_time=1.4, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(never, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.play(FadeIn(fine), run_time=0.6)

        # same target: with references, RL and cross-entropy aim at the same answer
        tgt_c = [0, -1.75, 0]
        target = VGroup(
            Circle(radius=0.42, stroke_color=INK2, stroke_width=2),
            Circle(radius=0.22, stroke_color=INK2, stroke_width=2),
            Dot(radius=0.08, color=INK),
        )
        target.move_to(tgt_c)
        tgt_l = txt("reference transcript", size=20, color=INK2).next_to(target, DOWN, buff=0.16)
        a_rl = Arrow(
            [-4.2, -1.2, 0], [-0.5, -1.7, 0], buff=0, stroke_width=4, color=INK2, tip_length=0.2
        )
        a_ce = Arrow(
            [4.2, -1.2, 0], [0.5, -1.7, 0], buff=0, stroke_width=4, color=INK2, tip_length=0.2
        )
        l_rl = txt("RL reward", size=22, color=INK).next_to(a_rl.get_start(), UP, buff=0.14)
        l_ce = txt("cross-entropy", size=22, color=INK).next_to(a_ce.get_start(), UP, buff=0.14)
        self.at(7, 0.3)
        self.play(FadeIn(target, scale=0.8), FadeIn(tgt_l), run_time=0.9, rate_func=SMOOTH)
        self.play(GrowArrow(a_rl), FadeIn(l_rl), run_time=1.0, rate_func=SMOOTH)
        self.play(GrowArrow(a_ce), FadeIn(l_ce), run_time=1.0, rate_func=SMOOTH)
        same = txt("same target, little left to teach", size=24, color=INK2).move_to([0, -3.05, 0])
        self.at(7, 10.0)
        self.play(FadeIn(same, shift=0.1 * UP), run_time=1.0, rate_func=SMOOTH)

        # ---- D: the verifier curve --------------------------------------------------------------
        x_lo, x_hi, y_lo, y_hi = 55.0, 100.0, 0.0, 30.0
        X0, X1, Y0, Y1 = -4.4, 5.6, -2.5, 1.4

        def P(acc: float, wer: float) -> np.ndarray:
            return np.array(
                [
                    X0 + (acc - x_lo) / (x_hi - x_lo) * (X1 - X0),
                    Y0 + (wer - y_lo) / (y_hi - y_lo) * (Y1 - Y0),
                    0,
                ]
            )

        title4 = txt("The limit: choosing without a reference", size=30, weight=SEMIBOLD).move_to(
            title
        )
        xa = Line(P(x_lo, 0), P(x_hi, 0), stroke_color=INK2, stroke_width=1.5)
        ya = Line(P(x_lo, 0), P(x_lo, y_hi), stroke_color=INK2, stroke_width=1.5)
        ticks = VGroup()
        for a in (60, 70, 80, 90, 100):
            ticks.add(Line(P(a, 0), P(a, 0) + [0, -0.07, 0], stroke_color=INK2, stroke_width=1.5))
            ticks.add(txt(str(a), size=18, color=MUTED).move_to(P(a, 0)).shift(0.28 * DOWN))
        for w in (10, 20, 30):
            ticks.add(
                Line(P(x_lo, w), P(x_lo, w) + [-0.07, 0, 0], stroke_color=INK2, stroke_width=1.5)
            )
            ticks.add(txt(str(w), size=18, color=MUTED).next_to(P(x_lo, w), LEFT, buff=0.16))
        x_t = (
            txt("candidate pairs ranked correctly (%)", size=20, color=INK2)
            .move_to(P(77.5, 0))
            .shift(0.65 * DOWN)
        )
        y_t = (
            txt("WER (%)", size=20, color=INK2)
            .next_to(P(x_lo, y_hi), UP, buff=0.14)
            .align_to(ya, LEFT)
        )
        frame = VGroup(xa, ya, ticks, x_t, y_t)

        noisy = VMobject(stroke_color=SYN, stroke_width=4).set_points_as_corners(
            [P(a, w) for a, w in curve("noisy oracle")]
        )
        blend = VMobject(stroke_color=SYN, stroke_width=3, stroke_opacity=0.55)
        blend.set_points_as_corners([P(a, w) for a, w in curve("blend")])
        n_l = txt("noisy oracle", size=20, color=INK2).next_to(P(66, 21), RIGHT, buff=0.2)
        b_l = (
            txt("blend toward oracle", size=20, color=INK2)
            .next_to(P(100, 4.91), UP, buff=0.18)
            .align_to(P(100, 0), RIGHT)
        )
        syn_l = (
            txt("simulated verifiers", size=22, color=INK)
            .next_to(n_l, UP, buff=0.12)
            .align_to(n_l, LEFT)
        )
        set_l = txt("development set, new sentences", size=18, color=MUTED).move_to(
            P(100, 30), aligned_edge=UP + RIGHT
        )

        o = ours("verifier_ours_ci.csv")
        e = ours("verifier_ours_ci_ctrl_ed.csv")

        def point(r: dict) -> VGroup:
            return VGroup(
                Line(
                    P(r["pair_acc_lo"], r["wer"]),
                    P(r["pair_acc_hi"], r["wer"]),
                    stroke_color=INK,
                    stroke_width=2,
                ),
                Line(
                    P(r["pair_acc"], r["wer_lo"]),
                    P(r["pair_acc"], r["wer_hi"]),
                    stroke_color=INK,
                    stroke_width=2,
                ),
                Dot(P(r["pair_acc"], r["wer"]), radius=0.11, color=INK),
            )

        pt = point(o)
        pt_l = (
            txt("our reranker", size=22, weight=SEMIBOLD)
            .next_to(pt, LEFT, buff=0.15)
            .shift(0.35 * DOWN)
        )

        self.at(8)
        self.play(FadeOut(VGroup(title3, chips, strikes, never, fine, target, tgt_l, a_rl, a_ce, l_rl, l_ce, same),
                          shift=0.15 * UP), run_time=0.9, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(title4, shift=0.1 * UP), FadeIn(frame), run_time=1.0, rate_func=SMOOTH)
        self.play(
            Create(noisy), FadeIn(syn_l), FadeIn(n_l), FadeIn(set_l), run_time=1.6, rate_func=SMOOTH
        )
        self.play(Create(blend), FadeIn(b_l), run_time=1.0, rate_func=SMOOTH)

        self.at(9)
        self.play(FadeIn(pt, shift=0.5 * DOWN), run_time=1.0, rate_func=SMOOTH)
        self.play(FadeIn(pt_l, shift=0.1 * RIGHT), run_time=0.6)
        span = Line(
            P(71, 0) + [0, 0.22, 0], P(78, 0) + [0, 0.22, 0], stroke_color=INK, stroke_width=6
        )
        span_l = txt("ours: 71–78% across sets", size=20, color=INK).next_to(span, LEFT, buff=0.2)
        self.play(Create(span), FadeIn(span_l, shift=0.08 * UP), run_time=1.0, rate_func=SMOOTH)

        # five more points of accuracy
        x5 = o["pair_acc"] + 5
        hits = [
            (83.25, 8.12),
            (83.27, 4.91),
        ]  # blend / noisy oracle at about +5 points (verifier_curve.csv)
        plus = Arrow(P(o["pair_acc"], o["wer"]), P(x5, o["wer"]), buff=0.15, stroke_width=4, color=INK,
                     max_tip_length_to_length_ratio=0.25, tip_length=0.18)  # fmt: skip
        plus_l = txt("+5 points", size=22, weight=SEMIBOLD).next_to(plus, UP, buff=0.1)
        drop = DashedLine(P(x5, o["wer"]) + [0, -0.12, 0], P(x5, hits[1][1]), dash_length=0.06, stroke_color=INK2,
                          stroke_width=2)  # fmt: skip
        hit_d = VGroup(*[Dot(P(a, w), radius=0.09, color=SYN) for a, w in hits])
        hit_l = VGroup(
            *[
                txt(f"{w:.1f}", size=20, color=INK).next_to(P(a, w), RIGHT, buff=0.16)
                for a, w in hits
            ]
        )
        gain = txt("13–50% lower WER", size=26, weight=SEMIBOLD).move_to(P(91, 17))
        self.at(10)
        self.play(
            FadeOut(span),
            FadeOut(span_l),
            GrowArrow(plus),
            FadeIn(plus_l),
            run_time=1.1,
            rate_func=SMOOTH,
        )
        self.play(Create(drop), LaggedStart(*[FadeIn(d, scale=0.5) for d in hit_d], lag_ratio=0.4),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(hit_l), FadeIn(gain, shift=0.1 * UP), run_time=0.9, rate_func=SMOOTH)

        # the encoder-adapted scorer moves us along the curve
        new = point(e)
        ring = Circle(radius=0.11, stroke_color=INK, stroke_width=2).move_to(pt[2])
        new_l = txt("+ encoder-adapted scorer", size=22, weight=SEMIBOLD).next_to(
            P(e["pair_acc"], e["wer"]), UR, buff=0.3
        )
        self.at(11)
        self.play(FadeOut(VGroup(plus, plus_l, drop, hit_d, hit_l, gain)), run_time=0.8)
        self.add(ring)
        self.play(
            pt.animate.become(new), pt_l.animate.set_opacity(0.5), run_time=1.4, rate_func=SMOOTH
        )
        self.play(FadeIn(new_l, shift=0.08 * UP), run_time=0.8, rate_func=SMOOTH)
        self.end()
