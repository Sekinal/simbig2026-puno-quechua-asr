"""S11: how many directions. A weight update splits into 16 directions; four of them keep the whole gain."""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    SEMIBOLD,
    UP,
    Arrow,
    Brace,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    Square,
    SurroundingRectangle,
    ValueTracker,
    VGroup,
    always_redraw,
)
from style import DEC, ENC, ENC_RAMP, GRID, INK, INK2, MUTED, SMOOTH, TalkScene, txt

# Directions kept -> energy share and WER on held-out sentences (reports/fig_data/rank_truncation.csv).
KEPT = [16, 8, 4, 2, 1]
ENERGY = {1: 60.6, 2: 74.4, 4: 84.9, 8: 93.4, 16: 100.0}
WER = {16: 16.43, 8: 16.52, 4: 16.35, 2: 17.04, 1: 17.83}
WER_LABEL = {16: "16.4", 8: "16.5", 4: "16.35", 2: "17.0", 1: "17.8"}
DEC_LORA = 20.2

# Per-direction energy shares, consistent with the cumulative shares above (illustrative split
# inside each block); strip heights show the singular value, i.e. the square root of the energy.
_E = [0.606, 0.138, 0.060, 0.045, 0.030, 0.022, 0.018, 0.015] + list(np.linspace(0.0105, 0.0060, 8))
SIGMA = np.sqrt(np.array(_E) / _E[0])

# Seeds (reports/RESULTS.md, exact average of the two enc+dec seeds, held-out sentences).
SEED_WER = ("16.0", "15.3", "15.65")


def update_matrix(n: int = 16, cell: float = 0.2) -> VGroup:
    """A weight update drawn as an n x n grid; shading follows a random low-rank matrix."""
    rng = np.random.default_rng(3)
    u, v = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    m = sum(SIGMA[k] ** 2 * np.outer(u[:, k], v[:, k]) for k in range(n))
    a = np.abs(m) / np.abs(m).max()
    cols = VGroup()
    for j in range(n):
        col = VGroup()
        for i in range(n):
            sq = Square(
                cell, stroke_width=0, fill_color=ENC, fill_opacity=0.08 + 0.82 * a[i, j] ** 0.8
            )
            sq.move_to([j * cell, -i * cell, 0])
            col.add(sq)
        cols.add(col)
    frame = Rectangle(width=n * cell, height=n * cell, stroke_color=INK2, stroke_width=1.5)
    frame.move_to(cols)
    return VGroup(cols, frame)


def fan(angles, scale: float, color: str, r: float = 1.05) -> VGroup:
    """Directions as arrows from the centre of a circle, lengths following the top singular values."""
    ring = Circle(radius=r, stroke_color=GRID, stroke_width=2)
    arrows = VGroup(
        *[
            Arrow(ORIGIN, scale * r * s * np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0]),
                  buff=0, stroke_width=5, color=color, max_tip_length_to_length_ratio=0.18, tip_length=0.16)
            for a, s in zip(angles, (1.0, 0.78, 0.62, 0.55), strict=False)
        ]
    )  # fmt: skip
    return VGroup(ring, arrows)


class S11(TalkScene):
    sid = "S11"

    def construct(self) -> None:
        self.intro_chip()

        # ---- the update matrix ------------------------------------------------------------------
        mat = update_matrix().move_to([-3.5, 0.0, 0])
        cols, frame = mat
        mat_label = txt("the update to one weight matrix", size=22, color=INK2).next_to(
            mat, UP, buff=0.3
        )
        self.at(0)
        self.play(
            FadeIn(mat, scale=0.96),
            FadeIn(mat_label, shift=0.1 * UP),
            run_time=1.2,
            rate_func=SMOOTH,
        )

        # ---- split into 16 directions, sorted by strength ---------------------------------------
        base_y, top_h, x0, dx = -1.8, 3.1, -6.0, 0.335
        targets = VGroup(
            *[
                Rectangle(
                    width=0.2,
                    height=top_h * SIGMA[j],
                    stroke_width=0,
                    fill_color=ENC,
                    fill_opacity=0.9,
                ).move_to([x0 + j * dx, base_y, 0], aligned_edge=DOWN)
                for j in range(16)
            ]
        )
        baseline = Line(
            [x0 - 0.25, base_y, 0],
            [x0 + 15 * dx + 0.25, base_y, 0],
            stroke_color=INK2,
            stroke_width=1.5,
        )
        dir_label = txt("16 directions, strongest first", size=22, color=INK2)
        dir_label.next_to(baseline, DOWN, buff=0.22).align_to(baseline, LEFT)
        self.at(1)
        self.play(
            *[cols[j].animate.move_to([x0 + j * dx, base_y + frame.height / 2 + 0.15, 0]) for j in range(16)],
            FadeOut(frame), FadeOut(mat_label, shift=0.1 * UP),
            run_time=1.1, rate_func=SMOOTH,
        )  # fmt: skip
        self.play(
            LaggedStart(*[ReplacementTransform(cols[j], targets[j]) for j in range(16)], lag_ratio=0.04),
            Create(baseline), FadeIn(dir_label, shift=0.1 * UP),
            run_time=1.5, rate_func=SMOOTH,
        )  # fmt: skip
        strips = targets

        e_val = ValueTracker(100)
        energy = always_redraw(
            lambda: txt(f"{e_val.get_value():.0f}%", size=40, weight=SEMIBOLD).move_to(
                [-3.75, 0.75, 0], aligned_edge=LEFT
            )
        )
        energy_cap = txt("energy kept", size=20, color=INK2)
        energy_cap.move_to([-3.75, 0.25, 0], aligned_edge=LEFT)
        energy_grp = VGroup(energy, energy_cap)

        # ---- WER vs directions kept -------------------------------------------------------------
        ax_l, ax_r, ax_b, ax_t = 0.9, 6.2, -1.8, 1.95
        y_lo, y_hi = 15.0, 21.0

        def px(k: int) -> float:
            return ax_r - (4 - np.log2(k)) / 4 * (ax_r - ax_l) if k else ax_l

        def py(w: float) -> float:
            return ax_b + (w - y_lo) / (y_hi - y_lo) * (ax_t - ax_b)

        x_axis = Line(
            [ax_l - 0.3, ax_b, 0], [ax_r + 0.3, ax_b, 0], stroke_color=INK2, stroke_width=1.5
        )
        y_axis = Line(
            [ax_l - 0.3, ax_b, 0], [ax_l - 0.3, ax_t, 0], stroke_color=INK2, stroke_width=1.5
        )
        ticks = VGroup(
            *[txt(str(k), size=20, color=INK2).move_to([px(k), ax_b - 0.28, 0]) for k in KEPT]
        )
        x_title = txt("directions kept", size=22, color=INK2).next_to(ticks, DOWN, buff=0.22)
        yticks = VGroup()
        for w in (16, 18, 20):
            yticks.add(
                Line(
                    [ax_l - 0.36, py(w), 0],
                    [ax_l - 0.3, py(w), 0],
                    stroke_color=INK2,
                    stroke_width=1.5,
                )
            )
            yticks.add(
                txt(str(w), size=18, color=MUTED).next_to([ax_l - 0.36, py(w), 0], LEFT, buff=0.1)
            )
        y_title = txt("WER, new sentences (%)", size=20, color=INK2)
        y_title.next_to([ax_l - 0.3, ax_t, 0], UP, buff=0.18).align_to(y_axis, LEFT)
        ref = DashedLine([ax_l - 0.3, py(DEC_LORA), 0], [ax_r + 0.3, py(DEC_LORA), 0],
                         dash_length=0.1, stroke_color=DEC, stroke_width=2.5)  # fmt: skip
        ref_label = (
            txt("decoder LoRA  20.2", size=20, color=INK2)
            .next_to(ref, DOWN, buff=0.12)
            .align_to(ref, RIGHT)
        )
        axes = VGroup(x_axis, y_axis, ticks, x_title, yticks, y_title)

        dot = Dot([px(16), py(WER[16]), 0], radius=0.11, color=ENC)
        labels = {16: txt(WER_LABEL[16], size=22, color=INK).next_to(dot, UP, buff=0.16)}
        self.play(FadeIn(axes), Create(ref), FadeIn(ref_label), run_time=1.1, rate_func=SMOOTH)
        self.play(FadeIn(dot, scale=0.5), FadeIn(labels[16], shift=0.1 * UP), FadeIn(energy_grp),
                  run_time=0.7, rate_func=SMOOTH)  # fmt: skip

        trail = VGroup()
        ghosts = VGroup()

        def keep(k: int, prev: int, run_time: float = 1.1) -> None:
            new = [dot.get_center(), [px(k), py(WER[k]), 0]]
            seg = Line(*new, stroke_color=ENC, stroke_width=3, stroke_opacity=0.55)
            ghost = Dot(new[0], radius=0.065, color=ENC, fill_opacity=0.55)
            trail.add(seg)
            ghosts.add(ghost)
            lab = txt(WER_LABEL[k], size=22, color=INK)
            lab.next_to(new[1], DOWN if k in (8,) else UP, buff=0.2)
            if k == 1:
                lab.shift(0.3 * RIGHT)
            labels[k] = lab
            self.add(ghost)
            self.play(
                *[strips[j].animate.set_fill(opacity=0.12) for j in range(k, prev)],
                e_val.animate.set_value(round(ENERGY[k])),
                dot.animate.move_to(new[1]), Create(seg),
                labels[prev].animate.set_opacity(0.55),
                run_time=run_time, rate_func=SMOOTH,
            )  # fmt: skip
            self.play(FadeIn(lab, shift=0.08 * UP), run_time=0.45, rate_func=SMOOTH)

        # 16 -> 8 -> 4: the whole gain survives
        self.at(2, 0.4)
        keep(8, 16)
        self.at(2, 2.6)
        keep(4, 8)

        # 16.35 versus 16.4
        self.at(3)
        hl = VGroup(
            SurroundingRectangle(labels[4], corner_radius=0.1, stroke_width=2, color=ENC, buff=0.1),
            SurroundingRectangle(
                labels[16], corner_radius=0.1, stroke_width=2, color=ENC, buff=0.1
            ),
        )
        self.play(
            labels[16].animate.set_opacity(1),
            Create(hl[0]),
            Create(hl[1]),
            run_time=0.9,
            rate_func=SMOOTH,
        )

        # 2 and 1: most of the gain is still there
        self.at(4)
        self.play(FadeOut(hl), run_time=0.4)
        keep(2, 4, run_time=0.9)
        keep(1, 2, run_time=0.9)

        # the top four directions do the work
        self.at(5, 0.3)
        brace = Brace(VGroup(*strips[:4]), DOWN, buff=0.1, color=INK2, sharpness=1.2)
        brace_lab = (
            txt("top 4 of 16", size=22, color=INK)
            .next_to(brace, DOWN, buff=0.12)
            .align_to(brace, LEFT)
        )
        self.play(
            *[strips[j].animate.set_fill(opacity=0.9) for j in range(1, 4)],
            e_val.animate.set_value(round(ENERGY[4])),
            dot.animate.move_to([px(4), py(WER[4]), 0]),
            labels[4].animate.set_opacity(1), labels[1].animate.set_opacity(0.55),
            FadeOut(dir_label),
            run_time=1.0, rate_func=SMOOTH,
        )  # fmt: skip
        self.play(
            FadeIn(brace, shift=0.05 * DOWN),
            FadeIn(brace_lab, shift=0.05 * DOWN),
            run_time=0.9,
            rate_func=SMOOTH,
        )

        self.at(6)
        cap = txt("The useful change is low-dimensional", size=30, t2c={"low-dimensional": ENC})
        cap.move_to([0, -3.1, 0])
        self.play(FadeIn(cap, shift=0.15 * UP), run_time=1.0, rate_func=SMOOTH)

        # ---- seeds: different directions, same result -------------------------------------------
        self.at(7)
        part1 = [m for m in self.content()]
        self.play(*[FadeOut(m, shift=0.15 * UP) for m in part1], run_time=0.9, rate_func=SMOOTH)

        cy = 0.35
        f1 = fan((15, 70, 128, 165), 0.92, ENC).move_to([-4.3, cy, 0])
        f2 = fan((205, 248, 300, 340), 0.92, ENC_RAMP[2]).move_to([-0.6, cy, 0])
        plus = txt("+", size=40, color=INK2).move_to([-2.45, cy, 0])
        n1 = txt("seed 1", size=24, weight=SEMIBOLD).next_to(f1, DOWN, buff=0.3)
        n2 = txt("seed 2", size=24, weight=SEMIBOLD).next_to(f2, DOWN, buff=0.3)
        w1 = txt(f"WER {SEED_WER[0]}", size=22, color=INK2).next_to(n1, DOWN, buff=0.12)
        w2 = txt(f"WER {SEED_WER[1]}", size=22, color=INK2).next_to(n2, DOWN, buff=0.12)
        overlap = txt("overlap of top 4:  0.02–0.08   (1 = identical)", size=20, color=MUTED)
        overlap.move_to([-2.45, -2.15, 0])
        self.play(FadeIn(f1[0]), LaggedStart(*[GrowArrow(a) for a in f1[1]], lag_ratio=0.15),
                  FadeIn(n1), FadeIn(w1), run_time=1.3, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(f2[0]), FadeIn(plus), LaggedStart(*[GrowArrow(a) for a in f2[1]], lag_ratio=0.15),
                  FadeIn(n2), FadeIn(w2), run_time=1.3, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(overlap, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        # exact average: half of each update
        self.at(8)
        eq = txt("=", size=40, color=INK2).move_to([1.45, cy, 0])
        ring3 = Circle(radius=1.05, stroke_color=GRID, stroke_width=2).move_to([3.6, cy, 0])
        c1, c2 = f1[1].copy(), f2[1].copy()
        self.play(FadeIn(eq), FadeIn(ring3), run_time=0.6, rate_func=SMOOTH)
        self.play(
            c1.animate.scale(0.5, about_point=f1[0].get_center()).shift(ring3.get_center() - f1[0].get_center()),
            c2.animate.scale(0.5, about_point=f2[0].get_center()).shift(ring3.get_center() - f2[0].get_center()),
            run_time=1.4, rate_func=SMOOTH,
        )  # fmt: skip
        n3 = (
            txt("mean of both", size=24, weight=SEMIBOLD)
            .next_to(ring3, DOWN, buff=0.3)
            .align_to(n1, DOWN)
        )
        w3 = txt(f"WER {SEED_WER[2]}", size=22, color=INK2).next_to(n3, DOWN, buff=0.12)
        note = txt("between the two: no gain", size=20, color=MUTED).next_to(w3, DOWN, buff=0.14)
        self.play(FadeIn(n3), FadeIn(w3), run_time=0.7, rate_func=SMOOTH)
        self.play(FadeIn(note, shift=0.08 * UP), run_time=0.7, rate_func=SMOOTH)

        self.at(9)
        cap2 = txt("Many equally good small changes", size=30, t2c={"Many": ENC})
        cap2.move_to([0, -2.95, 0])
        self.play(FadeIn(cap2, shift=0.15 * UP), run_time=1.0, rate_func=SMOOTH)
        self.end()
