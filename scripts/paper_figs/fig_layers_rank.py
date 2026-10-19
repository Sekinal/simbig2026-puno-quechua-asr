"""Figure: where inside the 7B encoder the gain lives (encoder-only LoRA, itest unseen).

(a) Which layers: the trained all-layer update is kept only in the shaded quarters of the 128
    layers and zeroed elsewhere (no retraining). Small diamonds: the same ablation on the two
    enc+dec adapters (decoder LoRA always kept). Rings: adapters retrained on one half only.
    The two rows that carry the headline (top / bottom quarter removed) sit on a light band.
(b) How many directions: every projection's update truncated to its top-k singular directions.
Reference lines: zero-shot, decoder-only LoRA (v2) and the full encoder adapter.
Right column: paired 95% interval of the difference to the full adapter, where computed (grey
when the interval includes 0).

Data: reports/fig_data/layer_ablation.csv, layer_ablation_diffs.csv, rank_truncation.csv,
where_to_adapt.csv (decoder-only LoRA of the 7B model).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import style as s
from matplotlib.patches import Rectangle

SET = "itest_unseen"
ENC = "encoder-only (7B)"
ED = ["enc+dec seed 0 (7B)", "enc+dec seed 1 (7B)"]
# layer rows: layers kept, y, comparison key in layer_ablation_diffs.csv (encoder-only adapter)
LAYER_ROWS = [
    ("0-127", 0.0, None),
    ("0-95", 1.35, "top quarter removed (0-95) - full"),
    ("32-127", 2.35, "bottom quarter removed (32-127) - full"),
    ("0-63", 3.7, "bottom half only (0-63) - full"),
    ("64-127", 4.7, None),
    ("0-31", 6.05, None),
    ("32-63", 7.05, None),
    ("64-95", 8.05, None),
    ("96-127", 9.05, None),
]
RANKS = [16, 8, 4, 2, 1]
XMIN, XMAX = 14.1, 29.0
# figure-fraction columns
SCH_L, SCH_W = 0.035, 0.175
AX_L, AX_W = 0.32, 0.46
TXT_X = 0.815
DOT, SMALL = s.DOT, s.DOT_SMALL
RETRAINED = {"0-63": "0-63", "64-127": "64-127"}  # rows that also show the retrained adapter
BAND = "#f4f3ef"  # light background band behind the two headline rows
GROUPS = [("all", 0.0), ("drop 1/4", 1.85), ("keep 1/2", 4.2), ("keep 1/4", 7.55)]  # (label, y)


def quarters(spec: str) -> set[int]:
    lo, hi = (int(v) for v in spec.split("-"))
    return {q for q in range(4) if q * 32 >= lo and q * 32 + 31 <= hi}


def cells(ax, y: float, n: int, filled: set[int], h: float = 0.62, gap: float = 0.1) -> None:
    """Draw a row of n cells in [0, 1] (axes x), the filled ones in the encoder colour."""
    w = (1 - gap * (n - 1) / n * 1) / n
    step = w + gap / n
    for i in range(n):
        on = i in filled
        ax.add_patch(
            Rectangle(
                (i * step, y - h / 2),
                w,
                h,
                facecolor=s.ENC if on else "#f1f0ec",
                edgecolor="none",
                zorder=2,
            )
        )


def ref_lines(ax, top: bool, zs: float, dec: float, full: float) -> None:
    for x, colour, ls, lw in (
        (zs, s.REF, (0, (2.5, 2)), 0.8),
        (dec, s.DEC, (0, (2.5, 2)), 0.8),
        (full, s.ENC, "-", 0.6),
    ):
        ax.axvline(x, color=colour, ls=ls, lw=lw, zorder=1)
    if top:
        kw = {
            "xycoords": ("data", "axes fraction"),
            "xytext": (0, 3),
            "textcoords": "offset points",
            "va": "bottom",
            "fontsize": s.FS_SMALL,
            "color": s.INK2,
            "annotation_clip": False,
        }
        # the full-adapter label hangs to the left of its line so it clears the decoder label
        for x, name, ha, dx in (
            (full, "full adapter", "right", 2),
            (dec, "decoder LoRA", "center", 0),
            (zs, "zero-shot", "center", 0),
        ):
            for i, line in enumerate(
                (name, s.fmt(x))
            ):  # usetex ignores linespacing: one text per line
                ax.annotate(line, (x, 1.0), ha=ha, **{**kw, "xytext": (dx, 3 + 8.5 * (1 - i))})


def delta(ax, y: float, r: dict | None) -> None:
    if r is None:
        return
    ax.annotate(
        rf"{s.signed(r['diff'])} {s.ci(r['lo'], r['hi'])}",
        (TXT_X, y),
        xycoords=("figure fraction", "data"),
        ha="left",
        va="center",
        fontsize=s.FS_SMALL,
        color=s.sig_colour(r["lo"], r["hi"]),
        annotation_clip=False,
    )


def main() -> None:
    s.setup()
    lab = s.read("layer_ablation.csv", fig_data=True)
    ldiff = s.read("layer_ablation_diffs.csv", fig_data=True).filter(pl.col("set") == SET)
    rank = s.read("rank_truncation.csv", fig_data=True)
    wta = s.read("where_to_adapt.csv", fig_data=True)
    dec = wta.filter((pl.col("model") == "omniASR-LLM-7B") & (pl.col("adapter") == "decoder"))[SET][
        0
    ]
    zs = lab.filter((pl.col("adapter") == ENC) & (pl.col("experiment") == "none"))[SET][0]
    kept = lab.filter(pl.col("experiment").is_in(["update kept after training", "all layers"]))

    def wer(adapter: str, layers: str) -> float | None:
        v = kept.filter((pl.col("adapter") == adapter) & (pl.col("layers_with_update") == layers))[
            SET
        ]
        return v[0] if len(v) else None

    full = wer(ENC, "0-127")
    retrained = lab.filter((pl.col("adapter") == ENC) & (pl.col("experiment") == "trained on half"))

    height = 5.9 * s.CM
    fig = plt.figure(figsize=(s.TEXTWIDTH, height))
    # vertical layout (figure fractions): block a on top, block b below, shared x axis at the bottom
    a_bot, a_h = 0.45, 0.415
    b_bot, b_h = 0.14, 0.215
    sch_a = fig.add_axes((SCH_L, a_bot, SCH_W, a_h))
    ax_a = fig.add_axes((AX_L, a_bot, AX_W, a_h))
    sch_b = fig.add_axes((SCH_L, b_bot, SCH_W, b_h))
    ax_b = fig.add_axes((AX_L, b_bot, AX_W, b_h))

    # ---- (a) which layers -------------------------------------------------------------------
    for ax in (sch_a, ax_a):  # the two headline rows: top / bottom quarter removed
        ax.axhspan(0.8, 2.9, color=BAND, lw=0, zorder=0)
    for layers, y, comp in LAYER_ROWS:
        cells(sch_a, y, 4, quarters(layers), gap=0.12)
        if layers in RETRAINED:
            v = retrained.filter(pl.col("layers_with_update") == layers)[SET][0]
            ax_a.scatter(
                [v],
                [y - 0.3],
                s=SMALL,
                facecolor="white",
                edgecolor=s.ENC,
                linewidths=0.9,
                zorder=5,
            )
        ax_a.scatter([wer(ENC, layers)], [y], s=DOT, color=s.ENC, linewidths=0, zorder=5)
        eds = [w for a in ED if (w := wer(a, layers)) is not None]
        if eds:
            ax_a.scatter(
                eds,
                [y + 0.3] * len(eds),
                s=SMALL,
                marker="D",
                facecolor="white",
                edgecolor=s.INK2,
                linewidths=0.6,
                zorder=4,
            )
        r = (
            ldiff.filter((pl.col("adapter") == ENC) & (pl.col("comparison") == comp)).row(
                0, named=True
            )
            if comp
            else None
        )
        delta(ax_a, y, r)
    for text, y in GROUPS:  # group names in the gutter, same column as panel b's energy share
        sch_a.text(1.05, y, text, ha="left", va="center", fontsize=s.FS_SMALL, color=s.MUTED)
    a_ylim = (10.55, -0.6)
    for x, ha, word in ((0.0, "left", "bottom"), (1.0, "right", "top")):
        sch_a.text(x, 9.6, word, ha=ha, va="top", fontsize=s.FS_SMALL, color=s.MUTED)
    # quarter boundaries above the schematic
    for i, lbl in enumerate(["0", "32", "64", "96", "128"]):
        sch_a.text(i / 4, -0.62, lbl, ha="center", va="bottom", fontsize=s.FS_SMALL, color=s.MUTED)
    sch_a.text(
        0.0,
        -1.72,
        "layers kept",
        ha="left",
        va="bottom",
        fontsize=s.FS_SMALL,
        color=s.INK2,
    )
    # ---- (b) how many singular directions ----------------------------------------------------
    for i, k in enumerate(RANKS):
        r = rank.filter(pl.col("directions_kept") == k).row(0, named=True)
        cells(sch_b, i, 16, set(range(k)), gap=0.25)
        sch_b.text(
            1.06,
            i,
            f"{round(r['energy_share'] * 100)}\\%",
            ha="left",
            va="center",
            fontsize=s.FS_SMALL,
            color=s.INK2,
        )
        ax_b.scatter([r[SET]], [i], s=DOT, color=s.ENC, linewidths=0, zorder=5)
        if r["lo"] is not None:
            delta(ax_b, i, {"diff": r["diff_itest_vs_full"], "lo": r["lo"], "hi": r["hi"]})
        sch_b.text(-0.04, i, str(k), ha="right", va="center", fontsize=s.FS_SMALL, color=s.INK2)
    b_ylim = (4.6, -0.6)
    sch_b.text(
        0.0, -0.95, "directions kept", ha="left", va="bottom", fontsize=s.FS_SMALL, color=s.INK2
    )
    sch_b.text(
        1.06, -0.95, "energy kept", ha="left", va="bottom", fontsize=s.FS_SMALL, color=s.INK2
    )

    # ---- shared styling ------------------------------------------------------------------------
    for sch, ylim in ((sch_a, a_ylim), (sch_b, b_ylim)):
        sch.set_xlim(0, 1)
        sch.set_ylim(*ylim)
        sch.axis("off")
    for ax, ylim, top in ((ax_a, a_ylim, True), (ax_b, b_ylim, False)):
        s.xgrid(ax)
        ref_lines(ax, top, zs, dec, full)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([15, 20, 25])
    ax_a.tick_params(axis="x", labelbottom=False, length=0)
    ax_a.spines["bottom"].set_visible(False)
    ax_b.set_xlabel(r"WER on held-out sentences (itest unseen, \%), $\leftarrow$ better")
    # column header and panel letters
    fig.text(
        TXT_X,
        a_bot + a_h + 0.012,
        "vs.\\ full adapter",
        ha="left",
        va="bottom",
        fontsize=s.FS_SMALL,
        color=s.MUTED,
    )
    sch_a.text(-0.2, -1.72, r"\textbf{a}", fontsize=s.FS, ha="left", va="bottom")
    sch_b.text(-0.2, -0.95, r"\textbf{b}", fontsize=s.FS, ha="left", va="bottom")
    s.save(fig, "layers_rank")


if __name__ == "__main__":
    main()
