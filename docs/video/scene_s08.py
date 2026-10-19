"""S08: it depends on the model (decoder vs encoder LoRA, dumbbell per model).

Data: reports/fig_data/where_to_adapt.csv, itest unseen WER. Whisper dots are the mean of seeds
0 and 1 (decoder 34.09 / 34.17, encoder 27.74 / 29.57); the seeds are the small satellite dots.
The out-of-domain note uses where_to_adapt_diffs.csv (Whisper encoder - decoder, OOD: -8.76, -7.63).
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
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
    SurroundingRectangle,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
)
from style import BG, DEC, ENC, GRID, INK, INK2, MUTED, SMOOTH, TalkScene, txt

W0, W1 = 14.0, 36.0  # WER range of the shared axis
X0, X1 = -2.7, 6.15  # scene x of that range
AXIS_Y = -2.5
LABEL_X = -3.15  # right edge of the row labels
R = 0.11  # dot radius
SEED_R = 0.05

ROWS = {  # name: (y, label, sub-label, decoder WER, encoder WER)
    "300M": (1.95, "omniASR 300M", None, 19.83, 25.39),
    "1B": (0.85, "omniASR 1B", None, 20.87, 21.57),
    "7B": (-0.25, "omniASR 7B", None, 20.17, 16.43),
    "W": (
        -1.5,
        "Whisper large-v3",
        ("zero-shot WER 141", "small dots: two seeds"),
        (34.09 + 34.17) / 2,
        (27.74 + 29.57) / 2,
    ),
}
WHISPER_SEEDS = {"dec": (34.09, 34.17), "enc": (27.74, 29.57)}
MATCHED = {"300M": (22.87, "rank 160"), "1B": (21.39, "rank 64")}  # param-matched encoder LoRA


def wx(w: float) -> float:
    return X0 + (w - W0) * (X1 - X0) / (W1 - W0)


def xw(x: float) -> float:
    return W0 + (x - X0) * (W1 - W0) / (X1 - X0)


def axis() -> VGroup:
    line = Line([X0, AXIS_Y, 0], [X1, AXIS_Y, 0], stroke_width=1.6, color=INK2)
    grid, ticks = VGroup(), VGroup()
    for w in (15, 20, 25, 30, 35):
        grid.add(Line([wx(w), AXIS_Y, 0], [wx(w), 2.45, 0], stroke_width=1, color=GRID))
        tick = Line([wx(w), AXIS_Y, 0], [wx(w), AXIS_Y - 0.08, 0], stroke_width=1.6, color=INK2)
        lab = txt(str(w), size=20, color=INK2).next_to(tick, DOWN, buff=0.1)
        ticks.add(tick, lab)
    title = txt("WER on held-out sentences (%)   ← better", size=20, color=INK2)
    title.move_to([(X0 + X1) / 2, AXIS_Y - 0.72, 0])
    return VGroup(grid, line, ticks, title)


def value(w: float, side, anchor, opacity: float = 1.0) -> VMobject:
    return txt(f"{w:.1f}", size=22).set_opacity(opacity).next_to(anchor, side, buff=0.12)


class Row:
    """One dumbbell: a decoder dot, an encoder dot that slides out of it, an arrow and values."""

    def __init__(self, key: str) -> None:
        y, label, sub, self.dec_w, self.enc_w = ROWS[key]
        self.y = y
        name = txt(label, size=24)
        self.label = VGroup(name)
        if sub:
            self.label.add(*[txt(t, size=17, color=MUTED) for t in sub])
            self.label.arrange(DOWN, buff=0.08, aligned_edge=RIGHT)
        self.label.move_to([0, y, 0]).align_to([LABEL_X, 0, 0], RIGHT)
        self.dec = Dot([wx(self.dec_w), y, 0], radius=R, color=DEC)
        self.enc = Dot([wx(self.dec_w), y, 0], radius=R, color=ENC)
        left = self.enc_w < self.dec_w  # encoder better: the arrow points left
        self.dec_val = value(self.dec_w, RIGHT if left else LEFT, self.dec)
        self.enc_side = LEFT if left else RIGHT

    def arrow_now(self) -> VMobject:
        a, b = self.dec.get_center(), self.enc.get_center()
        gap = abs(b[0] - a[0])
        if gap < 2 * R + 0.2:
            return Line(a, b, stroke_width=2.5, color=INK2).set_opacity(0 if gap < 2 * R else 1)
        return Arrow(a, b, buff=R + 0.03, stroke_width=3, color=INK2, tip_length=0.16,
                     max_tip_length_to_length_ratio=0.5, max_stroke_width_to_length_ratio=20)  # fmt: skip

    def show_decoder(self, scene: TalkScene, with_label: bool = True) -> None:
        anims = [GrowFromCenter(self.dec), FadeIn(self.dec_val, shift=0.1 * UP)]
        if with_label:
            anims.insert(0, FadeIn(self.label, shift=0.15 * RIGHT))
        scene.play(*anims, run_time=1.0, rate_func=SMOOTH)

    def slide_encoder(self, scene: TalkScene, run_time: float = 1.8) -> None:
        """The encoder dot leaves the decoder dot and slides to its WER, drawing the arrow."""
        self.arrow = always_redraw(self.arrow_now)
        op = ValueTracker(0)  # the value counts along with the dot and fades in on the way
        self.enc_val = always_redraw(
            lambda: value(xw(self.enc.get_x()), self.enc_side, self.enc, op.get_value())
        )
        scene.add(self.arrow, self.enc, self.enc_val)
        scene.bring_to_front(
            self.enc, self.dec
        )  # the encoder dot emerges from under the decoder dot
        scene.play(
            self.enc.animate.move_to([wx(self.enc_w), self.y, 0]),
            op.animate.set_value(1),
            run_time=run_time,
            rate_func=SMOOTH,
        )
        self.arrow.clear_updaters()
        self.enc_val.clear_updaters()

    def marks(self) -> VGroup:
        return VGroup(self.dec, self.enc, self.arrow, self.dec_val, self.enc_val)

    def enc_marks(self) -> VGroup:
        return VGroup(self.enc, self.arrow, self.enc_val)


class S08(TalkScene):
    sid = "S08"

    def construct(self) -> None:
        self.intro_chip()
        ax = axis()
        rows = {k: Row(k) for k in ROWS}
        r7, rw, r1, r3 = rows["7B"], rows["W"], rows["1B"], rows["300M"]

        # 0: same data, same recipe; decoder-only vs encoder-only LoRA
        self.at(0)
        self.play(
            Create(ax[1]),
            FadeIn(ax[0]),
            FadeIn(ax[2]),
            FadeIn(ax[3]),
            run_time=1.4,
            rate_func=SMOOTH,
        )

        # 1-2: the 7B row, 20.2 -> 16.4, with direct labels for the two dots
        self.at(1)
        r7.show_decoder(self)
        dec_lab = txt("decoder LoRA", size=20, color=INK).next_to(r7.dec, UP, buff=0.2)
        dec_lab.align_to(r7.dec, LEFT).shift(0.12 * LEFT)
        self.play(FadeIn(dec_lab, shift=0.1 * UP), run_time=0.7, rate_func=SMOOTH)
        r7.slide_encoder(self)
        enc_lab = txt("encoder LoRA", size=20, color=INK).next_to(r7.enc, UP, buff=0.2)
        enc_lab.align_to(r7.enc, RIGHT).shift(0.12 * RIGHT)
        self.play(FadeIn(enc_lab, shift=0.1 * UP), run_time=0.7, rate_func=SMOOTH)
        self.at(2)
        hl = RoundedRectangle(corner_radius=0.1, width=r7.marks().width + 0.4, height=0.6,
                              stroke_width=2, color=ENC).move_to(r7.marks())  # fmt: skip
        self.play(FadeIn(hl), run_time=0.6, rate_func=SMOOTH)
        self.wait(1.6)
        self.play(FadeOut(hl), run_time=0.6, rate_func=SMOOTH)

        # 3: Whisper, another family: the same direction; seeds as satellites
        self.at(3)
        sep = Line([X0 - 3.6, -0.88, 0], [X1, -0.88, 0], stroke_width=1, color=GRID)
        self.play(FadeIn(sep), FadeIn(rw.label, shift=0.15 * RIGHT),
                  run_time=0.8, rate_func=SMOOTH)  # fmt: skip
        seeds_dec = VGroup(*[Dot([wx(w), rw.y + dy, 0], radius=SEED_R, color=DEC)
                             for w, dy in zip(WHISPER_SEEDS["dec"], (0.22, -0.22), strict=True)])  # fmt: skip
        seeds_enc = VGroup(*[Dot([wx(w), rw.y + dy, 0], radius=SEED_R, color=ENC)
                             for w, dy in zip(WHISPER_SEEDS["enc"], (0.22, -0.22), strict=True)])  # fmt: skip
        rw.show_decoder(self, with_label=False)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in seeds_dec], lag_ratio=0.3), run_time=0.6)
        rw.slide_encoder(self, run_time=2.0)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in seeds_enc], lag_ratio=0.3), run_time=0.6)
        self.at(3, 5.6)  # "... and even more strongly out of domain"
        ood = txt("out of domain: 7.6–8.8 points better", size=18, color=INK2)
        ood.move_to([(rw.enc.get_x() + rw.dec.get_x()) / 2, rw.y - 0.47, 0])
        self.play(FadeIn(ood, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        # 4: the smaller omniASR encoders; direct labels move into a key first
        self.at(4)
        key_dec = VGroup(Dot(radius=R, color=DEC), txt("decoder LoRA", size=20)).arrange(
            RIGHT, buff=0.15
        )
        key_enc = VGroup(Dot(radius=R, color=ENC), txt("encoder LoRA", size=20)).arrange(
            RIGHT, buff=0.15
        )
        key = VGroup(key_dec, key_enc).arrange(RIGHT, buff=0.5).move_to([0, 2.95, 0])
        key.align_to([X1 + 0.25, 0, 0], RIGHT)
        self.play(
            dec_lab.animate.move_to(key_dec[1]),
            enc_lab.animate.move_to(key_enc[1]),
            FadeIn(key_dec[0]),
            FadeIn(key_enc[0]),
            run_time=1.2,
            rate_func=SMOOTH,
        )
        self.remove(dec_lab, enc_lab)
        self.add(key)
        self.play(FadeIn(r1.label, shift=0.15 * RIGHT), FadeIn(r3.label, shift=0.15 * RIGHT),
                  run_time=0.9, rate_func=SMOOTH)  # fmt: skip
        self.at(4, 4.2)  # "... with the very same decoder"
        self.play(
            LaggedStart(*[GrowFromCenter(r.dec) for r in (r1, r3)], lag_ratio=0.25),
            LaggedStart(*[FadeIn(r.dec_val, shift=0.1 * UP) for r in (r1, r3)], lag_ratio=0.25),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        self.at(4, 7.0)  # "... don't benefit": 1B barely moves
        r1.slide_encoder(self, run_time=1.4)

        # 5: for the 300M, encoder adaptation is clearly worse
        self.at(5)
        r3.slide_encoder(self, run_time=2.0)

        # 6: with the decoder adapted, all three omniASR models end up near 20
        self.at(6)
        omni = (r3, r1, r7)
        band = Rectangle(width=wx(21.25) - wx(19.45), height=2.95, stroke_width=0,
                         fill_color=DEC, fill_opacity=0.09)  # fmt: skip
        band.move_to([(wx(21.25) + wx(19.45)) / 2, 0.85, 0])
        band_lab = txt("≈ 20", size=22, color=INK).next_to(band, UP, buff=0.08)
        dim = [r.enc_marks() for r in omni] + [rw.marks(), seeds_dec, seeds_enc, ood, rw.label]
        self.add(band)
        self.bring_to_front(*[r.marks() for r in omni])
        band.set_opacity(0)
        self.play(
            *[m.animate.set_opacity(0.2) for m in dim],
            band.animate.set_fill(DEC, opacity=0.09),
            FadeIn(band_lab, shift=0.1 * DOWN),
            run_time=1.2,
            rate_func=SMOOTH,
        )

        # 7: only the 7B turns encoder adaptation into a further gain
        self.at(7)
        others = [r3.marks(), r1.marks(), r3.label, r1.label]
        self.play(
            FadeOut(band),
            FadeOut(band_lab),
            r7.enc_marks().animate.set_opacity(1),
            *[m.animate.set_opacity(0.2) for m in others],
            run_time=1.2,
            rate_func=SMOOTH,
        )
        hl7 = RoundedRectangle(corner_radius=0.1, width=r7.marks().width + 0.4, height=0.6,
                               stroke_width=2, color=ENC).move_to(r7.marks())  # fmt: skip
        self.play(FadeIn(hl7), run_time=0.6, rate_func=SMOOTH)

        # 8: is it adapter size? restore everything
        self.at(8)
        restore = [r.marks() for r in rows.values()] + [r.label for r in rows.values()]
        restore += [seeds_dec, seeds_enc, ood]
        self.play(
            FadeOut(hl7),
            *[m.animate.set_opacity(1) for m in restore],
            run_time=1.0,
            rate_func=SMOOTH,
        )

        # 9: a rank-16 adapter is ten times smaller on the small encoder
        self.at(9)
        cap = txt("rank-16 encoder LoRA:\n10× fewer parameters\non the 300M than on the 7B",
                  size=20, color=INK2, line_spacing=0.9)  # fmt: skip
        cap.move_to([4.55, 1.4, 0])
        self.play(FadeIn(cap, shift=0.1 * UP), *[m.animate.set_opacity(0.25) for m in (rw.marks(), rw.label,
                  seeds_dec, seeds_enc, ood, r7.marks(), r7.label)],
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip

        # 10: parameter-matched encoder adapters (hollow rings on a sub-row)
        self.at(10)
        rings, ring_vals = VGroup(), VGroup()
        for k, r in (("300M", r3), ("1B", r1)):
            w, rank = MATCHED[k]
            ring = Circle(radius=R, stroke_width=3, color=ENC).set_fill(BG, 1)
            ring.move_to(r.enc)
            rings.add(ring)
            v = VGroup(txt(f"{w:.1f}", size=22), txt(rank, size=17, color=MUTED)).arrange(
                RIGHT, buff=0.15
            )
            v.next_to([wx(w), r.y - 0.34, 0], RIGHT, buff=0.12 + R)
            ring_vals.add(v)
        key_ring = VGroup(
            Circle(radius=R, stroke_width=3, color=ENC), txt("param-matched", size=20)
        )
        key_ring.arrange(RIGHT, buff=0.15)
        self.play(
            key.animate.shift(LEFT * (key_ring.width + 0.5)),
            FadeOut(cap),
            run_time=0.8,
            rate_func=SMOOTH,
        )
        key_ring.next_to(key, RIGHT, buff=0.5)
        self.play(
            FadeIn(key_ring), *[GrowFromCenter(c) for c in rings], run_time=0.8, rate_func=SMOOTH
        )
        self.play(
            *[c.animate.move_to([wx(MATCHED[k][0]), r.y - 0.34, 0])
              for c, (k, r) in zip(rings, (("300M", r3), ("1B", r1)), strict=True)],
            run_time=1.6,
            rate_func=SMOOTH,
        )  # fmt: skip
        self.play(
            LaggedStart(*[FadeIn(v, shift=0.1 * RIGHT) for v in ring_vals], lag_ratio=0.2),
            run_time=0.8,
        )

        # 11: helps the 300M a little, but still behind the decoder
        self.at(11)
        guide = DashedLine([wx(19.83), r3.y - R, 0], [wx(19.83), r3.y - 0.5, 0], stroke_width=2, color=DEC,
                           dash_length=0.06)  # fmt: skip
        guide.set_opacity(0.6)
        hl3 = SurroundingRectangle(VGroup(r3.marks(), rings[0], ring_vals[0]), corner_radius=0.1, buff=0.12,
                                   stroke_width=2, color=INK2)  # fmt: skip
        self.play(Create(guide), FadeIn(hl3), r1.marks().animate.set_opacity(0.35), rings[1].animate.set_opacity(0.35),
                  ring_vals[1].animate.set_opacity(0.35), run_time=1.0, rate_func=SMOOTH)  # fmt: skip

        # 12: so it's not adapter capacity
        self.at(12)
        verdict = txt("not adapter capacity", size=28, color=INK).move_to([0, 1.25, 0])
        verdict.align_to([X1 + 0.25, 0, 0], RIGHT)
        self.play(FadeOut(hl3), FadeIn(verdict, shift=0.15 * UP),
                  *[m.animate.set_opacity(1) for m in (r1.marks(), rings[1], ring_vals[1])],
                  run_time=1.0, rate_func=SMOOTH)  # fmt: skip
        self.end()
