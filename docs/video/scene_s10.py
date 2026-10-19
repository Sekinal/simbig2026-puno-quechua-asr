"""S10: which layers of the 7B encoder matter (layer ablation of the encoder adapter).

Data: reports/fig_data/layer_ablation.csv, itest unseen WER unless noted.
* encoder-only adapter, update kept after training in: 0-127 16.43 (full), 0-95 16.17 (top quarter
  off), 32-127 20.09 (bottom quarter off), 32-63 23.22 (OOD 15.33); zero-shot 27.91 (OOD 23.26);
  full adapter OOD 14.95.
* trained on one half only: 0-63 16.09, 64-127 20.61.
* enc+dec seeds 0 / 1 (diamonds): 0-127 16.00 / 15.30, 0-95 16.09 / 15.39, 32-127 20.09 / 19.04.
* decoder-only LoRA 20.17 (reports/fig_data/where_to_adapt.csv), drawn as a reference.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Brace,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Square,
    TransformFromCopy,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
)
from style import BG, DEC, ENC, GRID, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt

OFF = "#e2e0d9"  # a layer whose update is switched off
W0, W1 = 14.0, 29.0  # shared WER axis
X0, X1 = -0.95, 6.2
AXIS_Y = -2.15
ROW_TOP = 2.2  # top of the reference lines
R = 0.11
ZS, DEC_W, FULL = 27.91, 20.17, 16.43  # zero-shot, decoder LoRA, full encoder adapter (itest)
ZS_OOD, FULL_OOD = 23.26, 14.95

# the big stack of 128 layers
SX, SW = -5.62, 0.42  # centre x, width
LX = -3.8  # left edge of the row labels
SY0, SY1 = -2.45, 2.25  # bottom (layer 0) and top (layer 127)
QGAP = 0.1
# per-row glyph (four cells, bottom quarter at the bottom)
GX, GW, GH, GGAP = -1.45, 0.17, 0.1, 0.025

ROWS = [  # key, y, label, quarters kept, start WER, end WER, enc+dec seed WERs
    ("full", 1.8, "full adapter", {0, 1, 2, 3}, ZS, FULL, (16.00, 15.30)),
    ("top", 1.1, "top ¼ off", {0, 1, 2}, FULL, 16.17, (16.09, 15.39)),
    ("bottom", 0.4, "bottom ¼ off", {1, 2, 3}, FULL, 20.09, (20.09, 19.04)),
    ("lo_half", -0.75, "bottom ½ only", {0, 1}, ZS, 16.09, None),
    ("hi_half", -1.45, "top ½ only", {2, 3}, ZS, 20.61, None),
]


def fmt(w: float) -> str:
    """One decimal, rounding halves up (14.95 -> 15.0, as in the paper)."""
    return f"{w + 1e-9:.1f}"


def wx(w: float) -> float:
    return X0 + (w - W0) * (X1 - X0) / (W1 - W0)


def xw(x: float) -> float:
    return W0 + (x - X0) * (W1 - W0) / (X1 - X0)


def label_txt(s: str, size: float = 22, color: str = INK) -> VMobject:
    """Text with a background halo so that it stays legible where it crosses a reference line."""
    t = txt(s, size=size, color=color)
    t.set_stroke(BG, width=6, background=True)
    return t


def quarter_box(q: int) -> tuple[float, float]:
    h = (SY1 - SY0 - 3 * QGAP) / 4
    lo = SY0 + q * (h + QGAP)
    return lo, lo + h


def stack() -> VGroup:
    """128 thin layers in four quarters (VGroup of four VGroups of 32)."""
    quarters = VGroup()
    for q in range(4):
        lo, hi = quarter_box(q)
        pitch = (hi - lo) / 32
        quarters.add(VGroup(*[
            Rectangle(width=SW, height=pitch * 0.62, stroke_width=0, fill_color=OFF, fill_opacity=1)
            .move_to([SX, lo + (j + 0.5) * pitch, 0])
            for j in range(32)
        ]))  # fmt: skip
    return quarters


def glyph(y: float, kept: set[int]) -> VGroup:
    """A small copy of the stack for one row: four cells, each made of 32 slivers (to morph into)."""
    cells = VGroup()
    total = 4 * GH + 3 * GGAP
    for q in range(4):
        lo = y - total / 2 + q * (GH + GGAP)
        pitch = GH / 32
        cells.add(VGroup(*[
            Rectangle(width=GW, height=pitch * 1.02, stroke_width=0, fill_color=ENC if q in kept else OFF,
                      fill_opacity=1).move_to([GX, lo + (j + 0.5) * pitch, 0])
            for j in range(32)
        ]))  # fmt: skip
    return cells


def light(st: VGroup, kept: set[int]) -> list:
    return [st[q].animate.set_fill(ENC if q in kept else OFF, 1) for q in range(4)]


def diamond(w: float, y: float) -> Square:
    return (
        Square(side_length=0.12, stroke_width=2, color=INK2)
        .set_fill(BG, 1)
        .rotate(0.785398)
        .move_to([wx(w), y, 0])
    )


def axis() -> VGroup:
    line = Line([X0, AXIS_Y, 0], [X1, AXIS_Y, 0], stroke_width=1.6, color=INK2)
    ticks = VGroup()
    for w in (15, 20, 25):
        t = Line([wx(w), AXIS_Y, 0], [wx(w), AXIS_Y - 0.08, 0], stroke_width=1.6, color=INK2)
        ticks.add(t, txt(str(w), size=20, color=INK2).next_to(t, DOWN, buff=0.1))
    title = txt("WER on held-out sentences (%)   ← better", size=20, color=INK2)
    title.move_to([(X0 + X1) / 2, AXIS_Y - 0.62, 0])
    return VGroup(line, ticks, title)


def ref_line(w: float, colour: str, dashed: bool, y0: float, y1: float) -> VMobject:
    if dashed:
        return DashedLine(
            [wx(w), y0, 0], [wx(w), y1, 0], stroke_width=2, color=colour, dash_length=0.07
        )
    return Line([wx(w), y0, 0], [wx(w), y1, 0], stroke_width=2, color=colour)


def ref_label(name: str, w: float, y: float) -> VGroup:
    g = VGroup(txt(name, size=18, color=INK2), txt(fmt(w), size=18, color=INK2)).arrange(
        DOWN, buff=0.06
    )
    return g.next_to([wx(w), y, 0], UP, buff=0.08)


class S10(TalkScene):
    sid = "S10"

    def slide(
        self, y: float, w_from: float, w_to: float, run_time: float = 1.6
    ) -> tuple[Dot, VMobject]:
        """A WER dot appears at w_from and slides to w_to; its value counts along."""
        dot = Dot([wx(w_from), y, 0], radius=R, color=ENC)
        op = ValueTracker(0)
        val = always_redraw(lambda: label_txt(fmt(xw(dot.get_x()))).set_opacity(op.get_value())
                            .next_to(dot, RIGHT, buff=0.12))  # fmt: skip
        self.play(GrowFromCenter(dot), run_time=0.5, rate_func=SMOOTH)
        self.add(val)
        self.bring_to_front(dot)
        self.play(dot.animate.move_to([wx(w_to), y, 0]), op.animate.set_value(1), run_time=run_time,
                  rate_func=SMOOTH)  # fmt: skip
        val.clear_updaters()
        return dot, val

    def construct(self) -> None:
        self.intro_chip()

        # 0: the 7B encoder, 128 layers in four quarters
        st = stack()
        st_title = txt("7B encoder", size=22).next_to([SX, SY1, 0], UP, buff=0.22)
        q_labels = VGroup()
        for q in range(4):
            lo, hi = quarter_box(q)
            q_labels.add(txt(f"{32 * q}–{32 * q + 31}", size=17, color=MUTED).next_to([SX - SW / 2, (lo + hi) / 2, 0],
                                                                                      LEFT, buff=0.1))  # fmt: skip
        io = txt("input", size=17, color=MUTED).next_to([SX, SY0, 0], DOWN, buff=0.12)
        self.at(0)
        self.play(FadeIn(st_title, shift=0.1 * UP),
                  LaggedStart(*[FadeIn(layer, shift=0.05 * UP) for layer in [m for q in st for m in q]], lag_ratio=0.01),
                  run_time=1.6, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(q_labels), FadeIn(io), run_time=0.7, rate_func=SMOOTH)

        # 1: switch the trained update off, one block at a time; axis and references
        self.at(1)
        ax = axis()
        refs = VGroup(
            ref_line(FULL, ENC, False, AXIS_Y, ROW_TOP),
            ref_line(DEC_W, DEC, True, AXIS_Y, ROW_TOP),
            ref_line(ZS, REF, True, AXIS_Y, ROW_TOP),
        )
        ref_labs = VGroup(
            ref_label("full adapter", FULL, ROW_TOP),
            ref_label("decoder LoRA", DEC_W, ROW_TOP),
            ref_label("zero-shot", ZS, ROW_TOP),
        )
        head1 = txt("update kept after training in", size=17, color=MUTED).move_to([0, 2.3, 0])
        head1.align_to([LX, 0, 0], LEFT)
        self.play(Create(ax[0]), FadeIn(ax[1:]), run_time=1.0, rate_func=SMOOTH)
        self.play(LaggedStart(*[Create(r) for r in refs], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(r, shift=0.1 * DOWN) for r in ref_labs], lag_ratio=0.2),
                  FadeIn(head1), run_time=1.4, rate_func=SMOOTH)  # fmt: skip

        rows = {}
        for key, y, lab, kept, w_from, w_to, seeds in ROWS:
            rows[key] = dict(y=y, kept=kept, w_from=w_from, w_to=w_to, seeds=seeds,
                             label=txt(lab, size=22).move_to([0, y, 0]).align_to([LX, 0, 0], LEFT),
                             glyph=glyph(y, kept))  # fmt: skip

        def show_row(key: str, slide_time: float = 1.6) -> None:
            r = rows[key]
            self.play(
                *light(st, r["kept"]),
                FadeIn(r["label"], shift=0.12 * RIGHT),
                run_time=0.8,
                rate_func=SMOOTH,
            )
            self.play(TransformFromCopy(st, r["glyph"]), run_time=0.9, rate_func=SMOOTH)
            r["dot"], r["val"] = self.slide(r["y"], r["w_from"], r["w_to"], slide_time)

        def row_group(key: str) -> VGroup:
            r = rows[key]
            return VGroup(r["label"], r["glyph"], r["dot"], r["val"])

        show_row("full")

        # 2: remove the top quarter: nothing changes
        self.at(2)
        show_row("top", 1.0)

        # 3-4: remove the bottom quarter: most of the gain is gone
        self.at(3)
        show_row("bottom", 1.8)
        self.at(4)
        hl = RoundedRectangle(corner_radius=0.1, width=wx(20.6) - wx(15.9) + 0.9, height=0.55, stroke_width=2,
                              color=ENC).move_to([(wx(20.6) + wx(15.9)) / 2 + 0.3, rows["bottom"]["y"], 0])  # fmt: skip
        self.play(FadeIn(hl), run_time=0.6, rate_func=SMOOTH)

        # 5: both enc+dec seeds show the same pattern (small diamonds under each row)
        self.at(5)
        dmds = VGroup()
        for key in ("full", "top", "bottom"):
            r = rows[key]
            dmds.add(*[diamond(w, r["y"] - 0.27) for w in r["seeds"]])
        key_d = VGroup(
            diamond(W0, 0), label_txt("enc+dec adapters, two seeds", size=17, color=INK2)
        ).arrange(RIGHT, buff=0.12)
        key_d.move_to([0, rows["full"]["y"] - 0.27, 0]).align_to([wx(17.6), 0, 0], LEFT)
        self.play(FadeOut(hl), LaggedStart(*[GrowFromCenter(d) for d in dmds], lag_ratio=0.12), run_time=1.6,
                  rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(key_d, shift=0.1 * RIGHT), run_time=0.7, rate_func=SMOOTH)

        # 6: adapters trained on one half only
        self.at(6)
        head2 = txt("adapter trained only on", size=17, color=MUTED).move_to([0, -0.3, 0])
        head2.align_to([LX, 0, 0], LEFT)
        sep = Line([LX, -0.12, 0], [X1, -0.12, 0], stroke_width=1, color=GRID)
        self.play(FadeIn(head2), FadeIn(sep), run_time=0.6, rate_func=SMOOTH)
        show_row("lo_half", 1.8)
        self.at(6, 4.4)  # "... while the top half alone gains nothing"
        show_row("hi_half", 1.6)

        # 7: the useful adaptation sits low, not where the update is biggest
        self.at(7)
        lo0, _ = quarter_box(0)
        _, hi1 = quarter_box(1)
        lo3, hi3 = quarter_box(3)
        b_use = Brace(
            Line([SX + SW / 2, lo0, 0], [SX + SW / 2, hi1, 0]),
            RIGHT,
            buff=0.08,
            color=INK2,
            sharpness=1.5,
        )
        b_big = Brace(
            Line([SX + SW / 2, lo3, 0], [SX + SW / 2, hi3, 0]),
            RIGHT,
            buff=0.08,
            color=MUTED,
            sharpness=1.5,
        )
        t_use = txt("useful", size=20).next_to(b_use, RIGHT, buff=0.08)
        t_big = VGroup(txt("largest", size=17, color=MUTED), txt("update", size=17, color=MUTED))
        t_big.arrange(DOWN, buff=0.05, aligned_edge=LEFT).next_to(b_big, RIGHT, buff=0.08)
        self.play(*light(st, {0, 1}), GrowFromCenter(b_use), FadeIn(t_use, shift=0.1 * LEFT),
                  GrowFromCenter(b_big), FadeIn(t_big, shift=0.1 * LEFT),
                  *[row_group(k).animate.set_opacity(0.3) for k in ("full", "hi_half")],
                  run_time=1.2, rate_func=SMOOTH)  # fmt: skip

        # 8: out of domain, the picture splits: only layers 32-63 kept
        self.at(8)
        old = VGroup(*[row_group(k) for k in rows], dmds, key_d, head1, head2, sep, refs, ref_labs,
                     b_use, t_use, b_big, t_big)  # fmt: skip
        self.play(FadeOut(old), *light(st, {1}), run_time=0.9, rate_func=SMOOTH)
        head3 = (
            txt("update kept only in layers 32–63", size=22)
            .move_to([0, 2.45, 0])
            .align_to([LX, 0, 0], LEFT)
        )
        tracks = {}
        for key, y, lab, zs, full in (("read", 1.2, ("held-out", "read speech"), ZS, FULL),
                                      ("radio", -0.45, ("radio drama", "out of domain"), ZS_OOD, FULL_OOD)):  # fmt: skip
            label = VGroup(txt(lab[0], size=22), txt(lab[1], size=22)).arrange(
                DOWN, buff=0.08, aligned_edge=LEFT
            )
            label.move_to([0, y, 0]).align_to([LX, 0, 0], LEFT)
            base = Line([X0, y, 0], [X1, y, 0], stroke_width=1.2, color=GRID)
            t_zs = ref_line(zs, REF, True, y - 0.3, y + 0.3)
            t_full = ref_line(full, ENC, False, y - 0.3, y + 0.3)
            l_zs = txt(f"zero-shot {fmt(zs)}", size=17, color=INK2).next_to(t_zs, UP, buff=0.06)
            l_full = txt(f"full adapter {fmt(full)}", size=17, color=INK2).next_to(
                t_full, UP, buff=0.06
            )
            tracks[key] = dict(y=y, zs=zs, group=VGroup(label, base, t_zs, t_full, l_zs, l_full))
        ax_title2 = txt("WER (%)   ← better", size=20, color=INK2).move_to(ax[2])
        self.play(
            FadeIn(head3, shift=0.1 * UP),
            FadeOut(ax[2]),
            FadeIn(ax_title2),
            run_time=0.6,
            rate_func=SMOOTH,
        )
        self.play(LaggedStart(*[FadeIn(tracks[k]["group"], shift=0.1 * UP) for k in ("read", "radio")],
                              lag_ratio=0.3), run_time=0.9, rate_func=SMOOTH)  # fmt: skip

        # 9: layers 32-63 recover the whole OOD gain; read speech also needs the bottom quarter
        self.at(9)
        self.slide(tracks["radio"]["y"], ZS_OOD, 15.33, 2.0)
        self.at(9, 4.6)
        self.slide(tracks["read"]["y"], ZS, 23.22, 1.8)
        lo0, hi0 = quarter_box(0)
        need = RoundedRectangle(corner_radius=0.06, width=SW + 0.14, height=hi0 - lo0 + 0.14, stroke_width=2,
                                color=INK2).move_to([SX, (lo0 + hi0) / 2, 0])  # fmt: skip
        need_t = VGroup(
            txt("read speech", size=17, color=INK2), txt("needs these", size=17, color=INK2)
        )
        need_t.arrange(DOWN, buff=0.05, aligned_edge=LEFT).next_to(need, RIGHT, buff=0.15)
        self.play(Create(need), FadeIn(need_t, shift=0.1 * LEFT), run_time=1.0, rate_func=SMOOTH)

        # 10: in-domain and out-of-domain effects live at different depths
        self.at(10)
        cap = txt("in-domain and out-of-domain gains live at different depths", size=24)
        cap.move_to([(X0 + X1) / 2 - 1.2, -1.45, 0])
        self.play(FadeIn(cap, shift=0.12 * UP), run_time=1.0, rate_func=SMOOTH)
        self.end()
