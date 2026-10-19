"""Figure: where to adapt. Itest-unseen WER with decoder-only vs encoder-only LoRA, per model.

Data: reports/fig_data/where_to_adapt.csv (WER, %) and where_to_adapt_diffs.csv (paired 95%
intervals of encoder - decoder; grey when the interval includes 0). Whisper has one row per
training seed; the parameter-matched encoder adapters (300M, 1B) are hollow. Whisper's zero-shot
output is not Quechua (WER 141), so it is off the scale. Arrows run decoder -> encoder, grey when
n.s., and are omitted when the two dots overlap (gap < 1 WER point).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import style as s

SET = "itest_unseen"
MODELS = [  # csv model, row label (None: sublabel only), seed, key in the diffs table, y
    ("omniASR-LLM-300M", "omniASR 300M", 0, "omniASR-LLM-300M", 0.0),
    ("omniASR-LLM-1B", "omniASR 1B", 0, "omniASR-LLM-1B", 1.0),
    ("omniASR-LLM-7B", "omniASR 7B", 0, "omniASR-LLM-7B", 2.0),
    ("Whisper-large-v3", None, 0, "Whisper-large-v3 (seed 0)", 3.2),
    ("Whisper-large-v3", None, 1, "Whisper-large-v3 (seed 1)", 3.8),
]
WHISPER_Y = 3.5  # centre of the two Whisper rows (row label)
XMIN, XMAX = 14, 36
DOT, RING = s.DOT, 22  # marker areas (pt^2)
ARROW_MIN = 1.0  # WER points; shorter encoder-decoder gaps get no arrow


def key(ax, x: float, y: float, kind: str, text: str) -> None:
    """Draw one entry of the key line (marker + ink text) at figure coordinates (x, y)."""
    tr = ax.figure.transFigure
    if kind == "tick":
        ax.plot(
            [x, x],
            [y - 0.028, y + 0.028],
            transform=tr,
            color=s.REF,
            lw=1.4,
            solid_capstyle="butt",
            clip_on=False,
        )
    elif kind == "ring":
        ax.scatter(
            [x],
            [y],
            s=RING,
            transform=tr,
            facecolor="white",
            edgecolor=s.ENC,
            linewidths=0.9,
            clip_on=False,
        )
    else:
        ax.scatter([x], [y], s=DOT, transform=tr, color=kind, linewidths=0, clip_on=False)
    ax.text(
        x + 0.014, y, text, transform=tr, ha="left", va="center", fontsize=s.FS_SMALL, color=s.INK
    )


def main() -> None:
    s.setup()
    df = s.read("where_to_adapt.csv", fig_data=True)
    diffs = s.read("where_to_adapt_diffs.csv", fig_data=True).filter(
        (pl.col("set") == SET) & (pl.col("comparison") == "encoder - decoder")
    )

    fig = plt.figure(figsize=(s.TEXTWIDTH, 4.5 * s.CM))
    ax = fig.add_axes((0.215, 0.185, 0.53, 0.655))
    s.xgrid(ax)

    for model, label, seed, dkey, y in MODELS:
        rows = df.filter((pl.col("model") == model) & (pl.col("seed") == seed))

        def vals(pred, rows: pl.DataFrame = rows) -> list[float]:
            return rows.filter(pred)[SET].to_list()

        dec = vals(pl.col("adapter") == "decoder")[0]
        enc = vals(pl.col("adapter") == "encoder")[0]
        zs = df.filter((pl.col("model") == model) & (pl.col("adapter") == "zero-shot"))[SET][0]
        matched = vals(pl.col("adapter").str.contains("param-matched"))

        if zs < XMAX:  # Whisper's (141) is off the scale; noted in the plot and the caption
            ax.plot(
                [zs] * 2,
                [y - 0.2, y + 0.2],
                color=s.REF,
                lw=1.4,
                solid_capstyle="butt",
                zorder=2,
            )
        r = diffs.filter(pl.col("model") == dkey).row(0, named=True)
        if abs(enc - dec) >= ARROW_MIN:  # no arrow between overlapping dots
            ax.annotate(
                "",
                xy=(enc, y),
                xytext=(dec, y),
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.32,head_width=0.16",
                    "color": s.INK2 if r["lo"] * r["hi"] > 0 else s.REF,  # n.s.: grey
                    "lw": 0.7,
                    "shrinkA": 3.4,
                    "shrinkB": 3.8,
                },
                zorder=3,
            )
        for v, colour in ((dec, s.DEC), (enc, s.ENC)):
            ax.scatter([v], [y], s=DOT, color=colour, linewidths=0, zorder=5)
        if matched:
            ax.scatter(
                matched,
                [y + 0.27],
                s=RING,
                facecolor="white",
                edgecolor=s.ENC,
                linewidths=0.9,
                zorder=4,
            )

        if label is not None:
            ax.text(
                -0.025,
                y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=s.FS,
                color=s.INK,
            )
        else:  # Whisper: one row per seed, labelled directly left of the encoder dot
            ax.annotate(
                f"seed {seed}",
                (enc, y),
                xytext=(-5, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=s.FS_SMALL,
                color=s.INK2,
            )
        ax.annotate(
            rf"{s.signed(r['diff'])} {s.ci(r['lo'], r['hi'])}",
            (1.04, y),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="center",
            fontsize=s.FS_SMALL,
            color=s.sig_colour(r["lo"], r["hi"]),
            annotation_clip=False,
        )

    for dy, text, size, colour in (
        (-0.2, "Whisper large-v3", s.FS, s.INK),
        (0.25, "zero-shot WER 141", s.FS_SMALL, s.MUTED),
    ):
        ax.text(
            -0.025,
            WHISPER_Y + dy,
            text,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=size,
            color=colour,
        )

    # column header and the key line, both above the plot
    ax.text(
        1.04,
        -0.75,
        r"encoder $-$ decoder",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=s.FS_SMALL,
        color=s.MUTED,
    )
    for x, kind, text in (
        (0.075, s.DEC, "decoder LoRA"),
        (0.25, s.ENC, "encoder LoRA"),
        (0.425, "ring", "encoder, param-matched to 7B"),
        (0.77, "tick", "zero-shot"),
    ):
        key(ax, x, 0.94, kind, text)
    # separator: Whisper is a different model family
    ax.plot([0, 1], [2.6, 2.6], transform=ax.get_yaxis_transform(), color=s.GRID, lw=0.6, zorder=1)

    ax.set_ylim(4.2, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([15, 20, 25, 30, 35])
    ax.set_xlabel(r"WER on held-out sentences (itest unseen, \%), $\leftarrow$ better")
    s.save(fig, "where_to_adapt")


if __name__ == "__main__":
    main()
