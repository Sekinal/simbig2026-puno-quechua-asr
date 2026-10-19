"""Figure: sentence reuse reverses the ranking of systems (slope chart, seen -> unseen sentences).

Data: reports/fig_data/sentence_reuse.csv (corpus WER, %). Every point is labelled, so the
chart carries no y-axis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import style as s

SYSTEMS = [  # csv name, direct label, colour, z-order, vertical label nudge (WER units, itest)
    (
        "CTC + word 4-gram on training prompts",
        "CTC + word 4-gram\nof training prompts",
        s.LM4,
        3,
        0.9,
    ),
    ("CTC greedy (no LM)", "CTC, no LM", s.REF, 2, -0.3),
    ("Hybrid + perturbed views (ours)", "ours", s.INK, 4, 0.0),
]
SPLITS = ["dev", "itest"]


def main() -> None:
    s.setup()
    df = s.read("sentence_reuse.csv", fig_data=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6 * s.CM, 4.3 * s.CM), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.63, bottom=0.10, top=0.88, wspace=0.62)
    for ax, split in zip(axes, SPLITS, strict=True):
        for name, label, colour, z, nudge in SYSTEMS:
            row = df.filter((df["system"] == name) & (df["split"] == split)).row(0, named=True)
            y = [row["seen"], row["unseen"]]
            ax.plot([0, 1], y, color=colour, lw=1.3, zorder=z)
            ax.scatter([0, 1], y, s=14, color=colour, zorder=z + 0.5, linewidths=0)
            kw = {
                "textcoords": "offset points",
                "va": "center",
                "fontsize": s.FS_SMALL,
                "color": s.INK2,
            }
            ax.annotate(f"{y[0]:.1f}", (0, y[0]), xytext=(-4, 0), ha="right", **kw)
            ax.annotate(f"{y[1]:.1f}", (1, y[1]), xytext=(4, 0), ha="left", **kw)
            if split == "itest":
                lines = label.split("\n")
                for i, line in enumerate(lines):  # one text per line: usetex ignores linespacing
                    dy = 4.3 * (len(lines) - 1) - 8.6 * i
                    ax.annotate(
                        line,
                        (1, y[1] + nudge),
                        xytext=(23, dy),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=s.FS_SMALL,
                        color=s.INK,
                        annotation_clip=False,
                    )
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 1], ["seen", "unseen"])
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.text(
            0.5,
            1.04,
            split,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=s.FS,
            color=s.INK,
        )
    axes[0].set_ylim(4, 27)
    s.save(fig, "sentence_reuse")


if __name__ == "__main__":
    main()
