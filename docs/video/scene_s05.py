"""S05: the system. Audio -> CTC and LLM-ASR branches -> candidates -> reranker, built left to right."""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    SEMIBOLD,
    UP,
    Animation,
    Create,
    CubicBezier,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    MoveToTarget,
    Rectangle,
    RoundedRectangle,
    Transform,
    VGroup,
    VMobject,
    config,
    there_and_back,
)
from style import BG, CARD, DEC, ENC, GRID, INK, INK2, MUTED, REF, SMOOTH, TalkScene, txt


def waveform(width=1.4, height=0.8, seed=3, n=90, noise=0.0, color=INK, stroke=2.4) -> VMobject:
    """A speech-like waveform: a few syllable envelopes times a fast oscillation."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    env = sum(
        a * np.exp(-(((x - c) / w) ** 2))
        for a, c, w in ((1.0, 0.22, 0.09), (0.7, 0.5, 0.08), (0.9, 0.76, 0.1))
    )
    env = 0.12 + env / env.max()
    y = env * np.sign(np.sin(np.arange(n) * 2.1)) * rng.uniform(0.45, 1.0, n)
    y += noise * rng.normal(0, 1, n)
    y = np.clip(y, -1, 1)
    pts = [np.array([(xi - 0.5) * width, yi * height / 2, 0]) for xi, yi in zip(x, y, strict=False)]
    w = VMobject(stroke_color=color, stroke_width=stroke)
    w.set_points_as_corners(pts)
    return w


def box(w, h, color=INK2, fill=CARD, fill_opacity=1.0, stroke=2.0) -> RoundedRectangle:
    return RoundedRectangle(width=w, height=h, corner_radius=0.14, stroke_color=color, stroke_width=stroke,
                            fill_color=fill, fill_opacity=fill_opacity)  # fmt: skip


def curve(a, b, color=GRID, width=2.5) -> CubicBezier:
    """Smooth horizontal-tangent connector from point a to point b."""
    a, b = np.array([*a, 0.0])[:3], np.array([*b, 0.0])[:3]
    dx = (b[0] - a[0]) * 0.5
    return CubicBezier(a, a + [dx, 0, 0], b - [dx, 0, 0], b, stroke_color=color, stroke_width=width)


def flow(path, color=INK2, n=3, run_time=1.6, lag=0.25, radius=0.055):
    """Dots travelling along a path, fading in and out (removed at the end)."""

    def one():
        d = Dot(path.point_from_proportion(0), radius=radius, color=color)

        def upd(m, a):
            m.move_to(path.point_from_proportion(SMOOTH(a)))
            m.set_opacity(min(1.0, 4 * a, 4 * (1 - a)))

        return _Along(d, upd)

    return LaggedStart(*[one() for _ in range(n)], lag_ratio=lag, run_time=run_time)


class _Along(Animation):
    def __init__(self, mob, fn, **kw):
        self.fn = fn
        super().__init__(mob, remover=True, **kw)

    def interpolate_mobject(self, alpha):
        self.fn(self.mobject, alpha)


def candidates(n=5, seed=0, color=REF, width=1.3) -> VGroup:
    """A stack of short rounded bars standing for candidate transcripts."""
    rng = np.random.default_rng(seed)
    bars = VGroup(*[
        RoundedRectangle(width=width * rng.uniform(0.62, 1.0), height=0.085, corner_radius=0.04,
                         stroke_width=0, fill_color=color, fill_opacity=1.0)
        for _ in range(n)
    ]).arrange(DOWN, buff=0.1, aligned_edge=LEFT)  # fmt: skip
    return bars


def patch(color, label="LoRA") -> VGroup:
    """A small glowing LoRA patch."""
    core = RoundedRectangle(width=0.78, height=0.32, corner_radius=0.1, stroke_color=color, stroke_width=2,
                            fill_color=color, fill_opacity=0.28)  # fmt: skip
    halo = VGroup(*[
        RoundedRectangle(width=0.78 + 0.07 * k, height=0.32 + 0.07 * k, corner_radius=0.1 + 0.035 * k,
                         stroke_width=0, fill_color=color, fill_opacity=0.07)
        for k in range(1, 5)
    ])  # fmt: skip
    t = txt(label, size=17, weight=SEMIBOLD).move_to(core)
    return VGroup(halo, core, t)


def check(color=INK, s=0.18) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=3.5)
    m.set_points_as_corners([[-s, 0, 0], [-s / 3, -s * 0.75, 0], [s, s * 0.8, 0]])
    return m


class S05(TalkScene):
    sid = "S05"

    def construct(self) -> None:
        self.intro_chip()

        # ---- layout --------------------------------------------------------------------------
        y_top, y_bot, y_mid = 1.7, -1.0, 0.3
        wave = waveform(width=1.35, height=0.95).move_to([-5.3, y_mid, 0])
        wave_lbl = txt("audio", size=20, color=MUTED).next_to(wave, DOWN, buff=0.3)
        src = np.array([-4.45, y_mid])

        ctc = box(2.5, 1.0).move_to([-1.75, y_top, 0])
        ctc_t = VGroup(
            txt("CTC  1B", size=26, weight=SEMIBOLD), txt("fine-tuned", size=18, color=INK2)
        )
        ctc_t.arrange(DOWN, buff=0.1).move_to(ctc)

        frame = RoundedRectangle(width=4.4, height=1.55, corner_radius=0.2, stroke_color=REF, stroke_width=1.5,
                                 fill_opacity=0).move_to([-1.75, y_bot, 0])  # fmt: skip
        frame_t = txt("omniASR-LLM  7B", size=20, color=INK2, weight=SEMIBOLD)
        frame_t.next_to(frame, UP, buff=0.12, aligned_edge=LEFT).shift(0.1 * RIGHT)
        enc = box(1.75, 0.95, color=ENC, fill=ENC, fill_opacity=0.1).move_to([-2.85, y_bot, 0])
        dec = box(1.75, 0.95, color=DEC, fill=DEC, fill_opacity=0.1).move_to([-0.65, y_bot, 0])
        enc_t = (
            VGroup(txt("speech", size=20), txt("encoder", size=20))
            .arrange(DOWN, buff=0.06)
            .move_to(enc)
        )
        dec_t = (
            VGroup(txt("LLaMA", size=20), txt("decoder", size=20))
            .arrange(DOWN, buff=0.06)
            .move_to(dec)
        )
        ed_arrow = Line(
            enc.get_right(), dec.get_left(), stroke_color=INK2, stroke_width=2.5, buff=0.04
        )
        ed_arrow.add_tip(tip_length=0.12, tip_width=0.12)
        p_enc = patch(ENC).move_to(enc.get_corner(UP + RIGHT) + [-0.35, 0.02, 0])
        p_dec = patch(DEC).move_to(dec.get_corner(UP + RIGHT) + [-0.35, 0.02, 0])

        x_cand = 0.75
        cand_top = candidates(5, seed=1).move_to([x_cand, y_top, 0], aligned_edge=LEFT)
        cand_bot = candidates(5, seed=2).move_to([x_cand, y_bot, 0], aligned_edge=LEFT)
        cand_top_t = VGroup(
            txt("16-best", size=17, color=MUTED), txt("word 4-gram", size=17, color=MUTED)
        )
        cand_top_t.arrange(DOWN, buff=0.06, aligned_edge=LEFT).next_to(cand_top, DOWN, buff=0.18)
        cand_top_t.align_to(cand_top, LEFT)
        cand_bot_t = (
            txt("N-best", size=17, color=MUTED)
            .next_to(cand_bot, DOWN, buff=0.18)
            .align_to(cand_bot, LEFT)
        )

        p_ctc_in = curve(src, ctc.get_left()[:2])
        p_enc_in = curve(src, frame.get_left()[:2])
        p_ctc_out = curve(ctc.get_right()[:2], [x_cand - 0.12, y_top])
        p_dec_out = curve(frame.get_right()[:2], [x_cand - 0.12, y_bot])

        # ---- 0: two branches -------------------------------------------------------------------
        self.at(0)
        self.play(Create(wave), FadeIn(wave_lbl, shift=0.1 * UP), run_time=1.2, rate_func=SMOOTH)
        self.play(Create(p_ctc_in), Create(p_enc_in), run_time=1.0, rate_func=SMOOTH)

        # ---- 1: CTC branch -> 16 candidates ----------------------------------------------------
        self.at(1)
        self.play(
            FadeIn(ctc, shift=0.15 * RIGHT),
            FadeIn(ctc_t, shift=0.15 * RIGHT),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        self.play(flow(p_ctc_in, n=3), run_time=1.6)
        self.play(Create(p_ctc_out), run_time=0.6, rate_func=SMOOTH)
        self.play(flow(p_ctc_out, n=3, run_time=1.0),
                  LaggedStart(*[GrowFromEdge(b, LEFT) for b in cand_top], lag_ratio=0.15, run_time=1.2),
                  run_time=1.4)  # fmt: skip
        self.play(FadeIn(cand_top_t, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        # ---- 2-3: LLM-ASR branch: encoder -> LLaMA decoder -------------------------------------
        self.at(2)
        self.play(Create(frame), FadeIn(frame_t, shift=0.1 * UP), run_time=1.2, rate_func=SMOOTH)
        self.play(flow(p_enc_in, n=3), run_time=1.6)
        self.at(3)
        self.play(LaggedStart(FadeIn(VGroup(enc, enc_t), shift=0.15 * RIGHT), Create(ed_arrow),
                              FadeIn(VGroup(dec, dec_t), shift=0.15 * RIGHT), lag_ratio=0.35),
                  run_time=1.5, rate_func=SMOOTH)  # fmt: skip
        self.play(Create(p_dec_out), run_time=0.6, rate_func=SMOOTH)
        self.play(flow(p_dec_out, n=3, run_time=1.0),
                  LaggedStart(*[GrowFromEdge(b, LEFT) for b in cand_bot], lag_ratio=0.15, run_time=1.2),
                  FadeIn(cand_bot_t), run_time=1.4)  # fmt: skip

        # ---- 4: LoRA patches glow ---------------------------------------------------------------
        self.at(4, 0.6)
        self.play(LaggedStart(FadeIn(p_enc, scale=0.8), FadeIn(p_dec, scale=0.8), lag_ratio=0.3),
                  run_time=1.2, rate_func=SMOOTH)  # fmt: skip
        halos = VGroup(p_enc[0], p_dec[0])
        self.play(halos.animate(rate_func=there_and_back).set_opacity(2.2), run_time=1.8)

        # ---- 5-7: self-training (an inset over a veiled pipeline) -------------------------------
        self.at(5)
        veil = Rectangle(width=config.frame_width, height=config.frame_height - 1.3, stroke_width=0,
                         fill_color=BG, fill_opacity=0.95).shift(0.25 * DOWN)  # fmt: skip
        st_title = txt("Self-training", size=34, weight=SEMIBOLD).move_to([0, 2.45, 0])
        self.play(FadeIn(veil), FadeIn(st_title, shift=0.15 * UP), run_time=1.0, rate_func=SMOOTH)

        self.at(6)
        rng = np.random.default_rng(7)
        tiles = VGroup(*[
            RoundedRectangle(width=0.36, height=0.22, corner_radius=0.05, stroke_color=REF, stroke_width=1.5,
                             fill_color=CARD, fill_opacity=1)
            for _ in range(24)
        ]).arrange_in_grid(rows=4, cols=6, buff=0.1).move_to([-4.3, 0.55, 0])  # fmt: skip
        pool_t = VGroup(
            txt("untranscribed", size=20, color=INK2),
            txt("spontaneous speech", size=20, color=INK2),
        )
        pool_t.arrange(DOWN, buff=0.06).next_to(tiles, DOWN, buff=0.25)
        names = ("LLM + LoRA", "CTC 1B", "silver labels")
        systems = VGroup(*[VGroup(box(1.75, 0.5), txt(n, size=18)) for n in names])
        for s in systems:
            s[1].move_to(s[0])
        systems.arrange(DOWN, buff=0.22).move_to([-0.9, 0.55, 0])
        vote = VGroup(box(1.2, 0.7, color=INK2), txt("vote", size=22, weight=SEMIBOLD))
        vote[1].move_to(vote[0])
        vote.move_to([1.65, 0.55, 0])
        kept = tiles.copy().move_to([4.4, 0.55, 0])
        keep_mask = rng.uniform(size=24) > 0.2
        kept_t = txt("27 h kept", size=26, weight=SEMIBOLD).next_to(kept, DOWN, buff=0.25)
        kept_s = txt("where they mostly agree", size=18, color=MUTED).next_to(
            kept_t, DOWN, buff=0.1
        )
        a_in = [curve(tiles.get_right()[:2], s[0].get_left()[:2], color=GRID) for s in systems]
        a_mid = [curve(s[0].get_right()[:2], vote[0].get_left()[:2], color=GRID) for s in systems]
        a_out = curve(vote[0].get_right()[:2], kept.get_left()[:2] + [-0.1, 0], color=GRID)

        self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in tiles], lag_ratio=0.03),
                  FadeIn(pool_t, shift=0.1 * UP), run_time=1.4, rate_func=SMOOTH)  # fmt: skip
        self.play(*[Create(a) for a in a_in], LaggedStart(*[FadeIn(s, shift=0.15 * RIGHT) for s in systems],
                  lag_ratio=0.2), run_time=1.4, rate_func=SMOOTH)  # fmt: skip
        self.play(*[flow(a, n=2, run_time=1.4) for a in a_in], run_time=1.4)
        self.play(
            *[Create(a) for a in a_mid],
            FadeIn(vote, shift=0.15 * RIGHT),
            run_time=1.2,
            rate_func=SMOOTH,
        )
        self.play(*[flow(a, n=2, run_time=1.4) for a in a_mid], run_time=1.4)
        self.play(Create(a_out), FadeIn(kept), run_time=1.0, rate_func=SMOOTH)
        self.play(*[
            t.animate.set_fill(INK2, opacity=0.85).set_stroke(INK2) if k else t.animate.set_opacity(0.15)
            for t, k in zip(kept, keep_mask, strict=False)
        ], flow(a_out, n=3, run_time=1.2), run_time=1.4, rate_func=SMOOTH)  # fmt: skip
        self.play(
            FadeIn(kept_t, shift=0.1 * UP),
            FadeIn(kept_s, shift=0.1 * UP),
            run_time=1.0,
            rate_func=SMOOTH,
        )

        # ---- 7: 27 h of pseudo-labels vs 5 h of real transcripts --------------------------------
        self.at(7)
        unit = 0.2  # scene units per hour
        b_real = Rectangle(
            width=5 * unit, height=0.32, stroke_width=0, fill_color=INK, fill_opacity=1
        )
        b_pseu = Rectangle(
            width=27 * unit, height=0.32, stroke_width=0, fill_color=INK2, fill_opacity=0.55
        )
        x0 = -2.2
        b_real.move_to([x0, -2.0, 0], aligned_edge=LEFT)
        b_pseu.move_to([x0, -2.55, 0], aligned_edge=LEFT)
        l_real = txt("real transcripts", size=20, color=INK2).next_to(b_real, LEFT, buff=0.25)
        l_pseu = txt("pseudo-labels", size=20, color=INK2).next_to(b_pseu, LEFT, buff=0.25)
        v_real = txt("5 h", size=20, weight=SEMIBOLD).next_to(b_real, RIGHT, buff=0.15)
        v_pseu = txt("27 h", size=20, weight=SEMIBOLD).next_to(b_pseu, RIGHT, buff=0.15)
        self.play(FadeIn(l_real), GrowFromEdge(b_real, LEFT), run_time=1.0, rate_func=SMOOTH)
        self.play(FadeIn(v_real), run_time=0.5)
        self.play(FadeIn(l_pseu), GrowFromEdge(b_pseu, LEFT), run_time=1.6, rate_func=SMOOTH)
        self.play(FadeIn(v_pseu), run_time=0.5)

        # back to the pipeline: the inset condenses into a data tag under the 7B branch
        self.at(7, 3.2)
        inset = VGroup(
            st_title, tiles, pool_t, systems, vote, kept, kept_t, kept_s, *a_in, *a_mid, a_out
        )
        bars = VGroup(b_real, b_pseu, l_real, l_pseu, v_real, v_pseu)
        tag = txt("data: 5 h real + 27 h pseudo-labels", size=17, color=MUTED)
        tag.next_to(frame, DOWN, buff=0.22)
        self.play(
            FadeOut(inset, shift=0.1 * UP),
            FadeOut(bars, shift=0.1 * UP),
            run_time=0.9,
            rate_func=SMOOTH,
        )
        self.play(FadeOut(veil), FadeIn(tag, shift=0.1 * UP), run_time=1.0, rate_func=SMOOTH)

        # ---- 8: six perturbed copies fan out ---------------------------------------------------
        self.at(8)
        ys = [y_mid + 0.43 * k for k in (3, 2, 1, 0, -1, -2, -3)]
        specs = [(1.05, 0.0, 11), (0.85, 0.0, 12), (1.0, 0.25, 13), None, (1.1, 0.0, 14), (0.95, 0.0, 15),
                 (1.0, 0.35, 16)]  # fmt: skip
        small = waveform(width=1.15, height=0.32).move_to([-5.3, y_mid, 0])
        copies = VGroup()
        for y, sp in zip(ys, specs, strict=False):
            if sp is None:
                continue
            wscale, noise, seed = sp
            c = waveform(
                width=1.15 / wscale, height=0.32, seed=seed, noise=noise, color=INK2, stroke=1.8
            )
            copies.add(c.move_to([-5.3, y, 0]))
        cap = VGroup(
            txt("6 perturbed copies", size=18, color=INK2),
            txt("faster · slower · noisier", size=17, color=MUTED),
        ).arrange(DOWN, buff=0.08)
        cap.move_to([-5.3, ys[-1] - 0.6, 0])
        self.play(Transform(wave, small), FadeOut(wave_lbl), run_time=1.0, rate_func=SMOOTH)
        self.play(LaggedStart(*[FadeIn(c, target_position=small) for c in copies], lag_ratio=0.12),
                  run_time=1.8, rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(cap, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)

        # ---- 9: each copy can surface a different transcript ------------------------------------
        self.at(9)
        joins = [
            curve([-4.65, c.get_y()], [src[0] + 0.02, y_mid], color=GRID, width=1.5) for c in copies
        ]
        self.play(*[Create(j) for j in joins], run_time=0.8, rate_func=SMOOTH)
        extra = candidates(3, seed=5).next_to(cand_bot, DOWN, buff=0.1, aligned_edge=LEFT)
        better = extra[1]
        cand_bot_t.generate_target().next_to(extra, DOWN, buff=0.18).align_to(cand_bot, LEFT)
        self.play(*[flow(j, n=1, run_time=1.0) for j in joins], run_time=1.0)
        self.play(flow(p_enc_in, n=4, run_time=1.4), run_time=1.4)
        self.play(flow(p_dec_out, n=3, run_time=1.0), MoveToTarget(cand_bot_t),
                  LaggedStart(*[GrowFromEdge(b, LEFT) for b in extra], lag_ratio=0.2, run_time=1.2),
                  run_time=1.4)  # fmt: skip
        self.play(better.animate.set_fill(INK), run_time=0.8, rate_func=SMOOTH)

        # ---- 10: reranker scores every candidate and picks one ---------------------------------
        self.at(10)
        rr = box(2.4, 2.25, color=INK, fill=CARD).move_to([3.95, 0.35, 0])
        rr_t = txt("reranker", size=26, weight=SEMIBOLD)
        terms = VGroup(
            txt("CTC likelihood", size=18, color=INK2),
            txt("+ LLM likelihoods", size=18, color=INK2),
            txt("+ character LM", size=18, color=INK2),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        VGroup(rr_t, terms).arrange(DOWN, buff=0.28).move_to(rr)
        p_rt = curve([x_cand + 1.42, y_top], rr.get_left()[:2] + [0, 0.45], color=GRID)
        p_rb = curve([x_cand + 1.42, y_bot - 0.2], rr.get_left()[:2] + [0, -0.45], color=GRID)
        self.play(
            FadeIn(rr, shift=0.15 * RIGHT),
            FadeIn(rr_t, shift=0.15 * RIGHT),
            run_time=1.0,
            rate_func=SMOOTH,
        )
        self.play(LaggedStart(*[FadeIn(t, shift=0.1 * RIGHT) for t in terms], lag_ratio=0.3),
                  Create(p_rt), Create(p_rb), run_time=1.5, rate_func=SMOOTH)  # fmt: skip
        self.at(10, 5.0)
        self.play(flow(p_rt, n=4, run_time=1.6), flow(p_rb, n=4, run_time=1.6), run_time=1.8)
        out = RoundedRectangle(width=0.7, height=0.12, corner_radius=0.05, stroke_width=0, fill_color=INK,
                               fill_opacity=1).move_to([6.0, 0.35, 0])  # fmt: skip
        p_out = Line(rr.get_right(), out.get_left(), stroke_color=INK2, stroke_width=2.5, buff=0.08)
        out_t = txt("1 transcript", size=18, color=INK2).next_to(out, DOWN, buff=0.2)
        ck = check(s=0.13).next_to(out, UP, buff=0.18)
        self.play(Create(p_out), GrowFromEdge(out, LEFT), run_time=1.0, rate_func=SMOOTH)
        self.play(FadeIn(out_t, shift=0.1 * UP), Create(ck), run_time=0.8, rate_func=SMOOTH)

        # ---- 11-12: tuning hygiene ------------------------------------------------------------
        self.at(11)
        tuned = txt("weights tuned on dev only", size=18, color=INK2).next_to(rr, DOWN, buff=0.25)
        self.play(FadeIn(tuned, shift=0.1 * UP), run_time=0.9, rate_func=SMOOTH)
        self.at(12)
        pills = VGroup()
        for name in ("internal test", "radio drama"):
            t = txt(name, size=18)
            p = RoundedRectangle(width=t.width + 0.4, height=0.42, corner_radius=0.21, stroke_color=INK2,
                                 stroke_width=1.5, fill_color=CARD, fill_opacity=1)  # fmt: skip
            pills.add(VGroup(p, t.move_to(p)))
        pills.arrange(RIGHT, buff=0.2).move_to([3.95, -2.25, 0])
        never = txt("never used for tuning", size=18, color=MUTED).next_to(pills, DOWN, buff=0.18)
        self.play(LaggedStart(*[FadeIn(p, shift=0.1 * UP) for p in pills], lag_ratio=0.25), run_time=1.0,
                  rate_func=SMOOTH)  # fmt: skip
        self.play(FadeIn(never, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.end()
