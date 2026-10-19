"""S09: same information, different plasticity (probes left, CKA vs depth right).

Data, read at render time:
* left: best-layer probe accuracy of each pretrained encoder (acc_zs, held-out speakers), from
  reports/probe_acc_300m.csv, probe_acc_1b.csv, probe_acc.csv (7B); shuffled-label controls are 46-55%.
* right: linear CKA pretrained vs adapted, dev frames, from reports/probe_cka_300m.csv,
  probe_cka_1b.csv, probe_cka.csv. 300M and 1B: encoder LoRA; 7B: the enc+dec adapter.
The scatter in the probe illustration is schematic (seeded random points), not data.
"""

from __future__ import annotations

import csv

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    Brace,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    VGroup,
    VMobject,
)
from style import ENC_RAMP, GRID, HERE, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt

REPORTS = HERE.parent.parent / "reports"
MODELS = [  # label, probe csv, cka csv, number of layers, colour
    ("300M", "probe_acc_300m.csv", "probe_cka_300m.csv", 24, ENC_RAMP[0]),
    ("1B", "probe_acc_1b.csv", "probe_cka_1b.csv", 48, ENC_RAMP[1]),
    ("7B", "probe_acc.csv", "probe_cka.csv", 128, ENC_RAMP[2]),
]
TASKS = [  # csv task, label lines
    ("ejective vs plain", ("ejective", "vs plain")),
    ("aspirated vs plain", ("aspirated", "vs plain")),
    ("q vs k (plain)", ("uvular q", "vs velar k")),
]

# left panel: horizontal bars, 0-100 %
BX0, BX1 = -4.3, -1.05  # scene x of 0 % and 100 %
GROUP_Y = (1.35, 0.0, -1.35)
BAR_DY, BAR_H = 0.32, 0.22
AX_Y = -2.0
# right panel: CKA (0.4-1.0) vs relative depth (0-1)
PX0, PX1, PY0, PY1 = 1.35, 5.25, -2.0, 2.0
C0, C1 = 0.4, 1.0


def read(name: str) -> list[dict]:
    with open(REPORTS / name) as f:
        return list(csv.DictReader(f))


def best_acc(name: str, task: str) -> float:
    return max(100 * float(r["acc_zs"]) for r in read(name) if r["task"] == task)


def cka_curve(name: str, n_layers: int) -> list[tuple[float, float]]:
    rows = [r for r in read(name) if r["set"] == "dev"]
    return sorted((int(r["layer"]) / (n_layers - 1), float(r["cka_zs_ed"])) for r in rows)


def bx(acc: float) -> float:
    return BX0 + acc / 100 * (BX1 - BX0)


def pt(depth: float, cka: float) -> np.ndarray:
    return np.array([PX0 + depth * (PX1 - PX0), PY0 + (cka - C0) / (C1 - C0) * (PY1 - PY0), 0])


def bar_y(group: int, model: int) -> float:
    return GROUP_Y[group] + (1 - model) * BAR_DY


def left_axis() -> VGroup:
    line = Line([BX0, AX_Y, 0], [BX1, AX_Y, 0], stroke_width=1.6, color=INK2)
    ticks = VGroup()
    for a in (0, 50, 100):
        t = Line([bx(a), AX_Y, 0], [bx(a), AX_Y - 0.08, 0], stroke_width=1.6, color=INK2)
        ticks.add(t, txt(str(a), size=20, color=INK2).next_to(t, DOWN, buff=0.1))
    title = txt("probe accuracy (%)", size=20, color=INK2).move_to(
        [(BX0 + BX1) / 2, AX_Y - 0.62, 0]
    )
    return VGroup(line, ticks, title)


def right_axes() -> VGroup:
    grid, ylabs = VGroup(), VGroup()
    for c in (0.4, 0.6, 0.8, 1.0):
        grid.add(Line(pt(0, c), pt(1, c), stroke_width=1, color=GRID))
        ylabs.add(txt(f"{c:.1f}", size=20, color=INK2).next_to(pt(0, c), LEFT, buff=0.15))
    xline = Line(pt(0, C0), pt(1, C0), stroke_width=1.6, color=INK2)
    yline = Line(pt(0, C0), pt(0, C1 + 0.02), stroke_width=1.6, color=INK2)
    xlabs = VGroup(
        txt("input", size=20, color=INK2).next_to(pt(0, C0), DOWN, buff=0.15),
        txt("output", size=20, color=INK2).next_to(pt(1, C0), DOWN, buff=0.15),
        txt("relative depth", size=20, color=INK2).next_to(pt(0.5, C0), DOWN, buff=0.15),
    )
    return VGroup(grid, xline, yline, ylabs, xlabs)


def probe_sketch() -> VGroup:
    """Schematic: two clouds of frames, ejective (filled) and plain (hollow), and a separating line."""
    rng = np.random.default_rng(3)
    centre = np.array([3.3, -0.1, 0])
    filled, hollow = VGroup(), VGroup()
    for _ in range(26):
        p = centre + np.array(
            [*np.clip(rng.normal([-0.75, 0.55], 0.32), [-1.4, -0.1], [-0.1, 1.2]), 0]
        )
        filled.add(Dot(p, radius=0.055, color=INK2))
        q = centre + np.array(
            [*np.clip(rng.normal([0.75, -0.55], 0.32), [0.1, -1.2], [1.4, 0.1]), 0]
        )
        hollow.add(Circle(radius=0.055, stroke_width=2, color=INK2).move_to(q))
    line = Line(
        centre + np.array([-1.3, -1.45, 0]),
        centre + np.array([1.3, 1.45, 0]),
        stroke_width=3,
        color=INK,
    )
    lab_e = txt("ejective", size=20, color=INK2).move_to(centre + np.array([-1.45, 1.35, 0]))
    lab_p = txt("plain", size=20, color=INK2).move_to(centre + np.array([1.45, -1.35, 0]))
    title = txt("a linear probe", size=24).move_to(centre + np.array([0, 2.15, 0]))
    note = txt("schematic", size=17, color=MUTED).next_to(title, DOWN, buff=0.1)
    return VGroup(filled, hollow, line, lab_e, lab_p, title, note)


class S09(TalkScene):
    sid = "S09"

    def construct(self) -> None:
        self.intro_chip()

        # ---- left panel: probes ------------------------------------------------------------
        title_l = txt("What the pretrained encoders know", size=24).move_to([-3.45, 2.85, 0])
        sub_l = txt("linear probes, best layer, held-out speakers", size=17, color=MUTED)
        sub_l.next_to(title_l, DOWN, buff=0.12)
        lax = left_axis()
        task_labels = VGroup()
        for g, (_, lines) in enumerate(TASKS):
            lab = VGroup(*[txt(t, size=20) for t in lines]).arrange(
                DOWN, buff=0.06, aligned_edge=RIGHT
            )
            task_labels.add(lab.move_to([0, GROUP_Y[g], 0]).align_to([-5.1, 0, 0], RIGHT))
        bars, model_labels = VGroup(), VGroup()
        for g, (task, _) in enumerate(TASKS):
            for m, (name, acc_csv, _, _, colour) in enumerate(MODELS):
                y = bar_y(g, m)
                acc = best_acc(acc_csv, task)
                bars.add(Rectangle(width=bx(acc) - BX0, height=BAR_H, stroke_width=0, fill_color=colour,
                                   fill_opacity=1).move_to([(bx(acc) + BX0) / 2, y, 0]))  # fmt: skip
                model_labels.add(
                    txt(name, size=17, color=MUTED).next_to([BX0, y, 0], LEFT, buff=0.1)
                )
        top = bar_y(0, 0) + 0.3
        control = DashedLine(
            [bx(50), AX_Y, 0], [bx(50), top, 0], stroke_width=2.5, color=REF, dash_length=0.08
        )
        control_lab = txt("shuffled-label control", size=18, color=INK2).next_to(
            control, UP, buff=0.08
        )

        self.at(0)
        self.play(FadeIn(title_l, shift=0.1 * UP), run_time=0.9, rate_func=SMOOTH)
        self.play(FadeIn(sub_l), run_time=0.6, rate_func=SMOOTH)
        self.at(1)
        self.play(Create(lax[0]), FadeIn(lax[1:]), run_time=1.2, rate_func=SMOOTH)
        self.at(2)
        self.play(LaggedStart(*[FadeIn(t, shift=0.15 * RIGHT) for t in task_labels], lag_ratio=0.45),
                  run_time=3.6, rate_func=SMOOTH)  # fmt: skip
        self.at(3)
        self.play(FadeIn(model_labels), run_time=0.6, rate_func=SMOOTH)
        self.play(
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.08),
            run_time=2.4,
            rate_func=SMOOTH,
        )
        self.play(
            Create(control), FadeIn(control_lab, shift=0.1 * DOWN), run_time=1.2, rate_func=SMOOTH
        )

        # 4: the information is there at every size
        self.at(4)
        brace = Brace(VGroup(*[b for b in bars]), RIGHT, buff=0.12, color=INK2, sharpness=1.5)
        rng_lab = txt("94–97 %", size=30, weight=SEMIBOLD).next_to(brace, RIGHT, buff=0.12)
        self.play(
            GrowFromCenter(brace), FadeIn(rng_lab, shift=0.1 * LEFT), run_time=1.0, rate_func=SMOOTH
        )
        left = VGroup(
            sub_l, lax, task_labels, bars, model_labels, control, control_lab, brace, rng_lab
        )

        # 5-6: what a probe is
        self.at(5)
        sketch = probe_sketch()
        filled, hollow, sep, lab_e, lab_p, s_title, s_note = sketch
        self.play(FadeIn(s_title, shift=0.1 * UP), FadeIn(s_note), run_time=0.8, rate_func=SMOOTH)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in [*filled, *hollow]], lag_ratio=0.02),
                  FadeIn(lab_e), FadeIn(lab_p), run_time=1.8, rate_func=SMOOTH)  # fmt: skip
        self.at(6)
        self.play(Create(sep), run_time=1.4, rate_func=SMOOTH)

        # 7: what differs is how far adaptation moves the encoder
        self.at(7)
        title_r = txt("How far adaptation moves them", size=24).move_to(
            [(PX0 + PX1) / 2 + 0.2, 2.85, 0]
        )
        sub_r = txt("linear CKA, pretrained vs adapted encoder", size=17, color=MUTED)
        sub_r.next_to(title_r, DOWN, buff=0.12)
        rax = right_axes()
        self.play(FadeOut(sketch), left.animate.set_opacity(0.3), title_l.animate.set_opacity(0.3),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(title_r, shift=0.1 * UP), FadeIn(sub_r), Create(rax[1]), Create(rax[2]), FadeIn(rax[0]),
                  FadeIn(rax[3:]), run_time=1.2, rate_func=SMOOTH)  # fmt: skip

        curves, ends = {}, {}
        for name, _, cka_csv, n_layers, colour in MODELS:
            pts = [pt(d, c) for d, c in cka_curve(cka_csv, n_layers)]
            curve = VMobject(stroke_width=3.5, color=colour).set_points_as_corners(pts)
            curves[name] = curve
            end = txt(name, size=20)
            if name == "7B":  # footnote: the 7B curve comes from the enc+dec adapter
                end = VGroup(end, txt("enc+dec", size=17, color=MUTED)).arrange(
                    DOWN, buff=0.06, aligned_edge=LEFT
                )
            ends[name] = end.next_to(pts[-1], RIGHT, buff=0.12)

        # 8: the small encoders barely change except in their last layers
        self.at(8)
        self.play(Create(curves["300M"]), Create(curves["1B"]), run_time=3.2, rate_func=SMOOTH)
        self.play(FadeIn(ends["300M"]), FadeIn(ends["1B"]), run_time=0.6, rate_func=SMOOTH)

        # 9: the 7B reorganizes from its lower-middle layers on (enc+dec adapter)
        self.at(9)
        self.play(Create(curves["7B"]), run_time=2.6, rate_func=SMOOTH)
        drop = cka_curve("probe_cka.csv", 128)[2]  # layer 16 of 128
        mark = Dot(pt(*drop), radius=0.07, color=INK)
        mark_lab = txt(f"{drop[1]:.2f} by layer 16 of 128", size=20).move_to(pt(0.4, 0.91))
        mark_lab.align_to(pt(0.17, 0), LEFT)
        self.play(FadeIn(ends["7B"]), GrowFromCenter(mark), FadeIn(mark_lab, shift=0.1 * DOWN),
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip

        # 10-11: what CKA measures
        self.at(10)
        explain = VGroup(txt("CKA = 1:  unchanged", size=20, color=INK2),
                         txt("lower:  the encoder moved", size=20, color=INK2))  # fmt: skip
        explain.arrange(DOWN, buff=0.14, aligned_edge=LEFT).move_to(pt(0.05, 0.6)).align_to(
            pt(0.05, 0), LEFT
        )
        self.play(FadeIn(explain[0], shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.at(11)
        self.play(FadeIn(explain[1], shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        # 12: same information, different plasticity
        self.at(12)
        cap_l = txt("Same information,", size=34, weight=SEMIBOLD).move_to(title_l)
        cap_r = txt("different plasticity", size=34, weight=SEMIBOLD).move_to(title_r)
        title_l.set_opacity(1)
        self.play(left.animate.set_opacity(1), FadeOut(title_l, shift=0.2 * UP), FadeIn(cap_l, shift=0.2 * UP),
                  run_time=1.1, rate_func=SMOOTH)  # fmt: skip
        self.play(
            FadeOut(title_r, shift=0.2 * UP),
            FadeIn(cap_r, shift=0.2 * UP),
            run_time=1.1,
            rate_func=SMOOTH,
        )

        # 13-14: a working hypothesis, not a proof
        self.at(14)
        hyp = txt("a working hypothesis, not a proof", size=20, color=MUTED).move_to([0, -3.2, 0])
        self.play(FadeIn(hyp, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.end()
