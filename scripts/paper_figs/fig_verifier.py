"""Figure: how good must a verifier be? Selection WER vs pairwise ranking accuracy.

Two synthetic verifier families on the perturbed-view pool (clips <= 39 s): our hybrid blended
toward the oracle, and a noisy oracle (negative error plus Gaussian noise); our hybrid's operating
point with 95% intervals on both axes. One panel per evaluation set, shared y axis.

Data: reports/verifier_curve.csv, reports/verifier_ours_ci.csv (scripts/verifier_curve.py).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import style as s

SETS = [
    ("dev_unseen", "dev unseen"),
    ("itest_unseen", "itest unseen"),
    ("dev_spont", "dev spontaneous"),
    ("ood", "OOD radio drama"),
]
FAMILIES = [("blend", "-"), ("noisy oracle", (0, (3, 1.6)))]  # family, line style


def main() -> None:
    s.setup()
    curve = s.read("verifier_curve.csv")
    ours = s.read("verifier_ours_ci.csv")
    fig, axes = plt.subplots(1, 4, figsize=(s.TEXTWIDTH, 4.1 * s.CM), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.235, top=0.86, wspace=0.12)
    for j, (ax, (key, title)) in enumerate(zip(axes, SETS, strict=True)):
        s.ygrid(ax)
        for fam, ls in FAMILIES:
            d = curve.filter((pl.col("set") == key) & (pl.col("family") == fam)).sort("pair_acc")
            ax.plot(d["pair_acc"], d["wer"], color=s.SYN, ls=ls, lw=1.1, zorder=3)
        o = ours.filter(pl.col("set") == key).row(0, named=True)
        ax.plot([o["pair_acc_lo"], o["pair_acc_hi"]], [o["wer"]] * 2, color=s.INK, lw=0.7, zorder=4)
        ax.plot([o["pair_acc"]] * 2, [o["wer_lo"], o["wer_hi"]], color=s.INK, lw=0.7, zorder=4)
        ax.scatter([o["pair_acc"]], [o["wer"]], s=20, color=s.INK, linewidths=0, zorder=5)
        ax.text(
            0.5,
            1.04,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=s.FS_SMALL,
            color=s.INK,
        )
        kw = {
            "textcoords": "offset points",
            "fontsize": 6.5,
            "color": s.INK2,
            "ha": "left",
            "va": "center",
        }
        if j == 0:  # direct labels, once each, where there is room
            ax.annotate("ours", (o["pair_acc"], o["wer"]), xytext=(5, 6), **kw)
        if j == 1:
            ax.annotate("noisy oracle", (66.5, 30), xytext=(0, 0), **kw)
            ax.annotate("blend to", (85, 19.5), xytext=(0, 0), **kw)
            ax.annotate("oracle", (85, 19.5), xytext=(0, -7.5), **kw)
        if j > 0:  # the grid carries the shared y scale
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)
        ax.set_xlim(55, 102)
        ax.set_xticks([60, 80, 100])
    axes[0].set_ylim(0, 35)
    axes[0].set_yticks([0, 10, 20, 30])
    axes[0].set_ylabel(r"WER (\%)")
    fig.text(
        0.535,
        0.02,
        r"verifier pairwise ranking accuracy (\%)",
        ha="center",
        va="bottom",
        fontsize=s.FS,
        color=s.INK2,
    )
    s.save(fig, "verifier")


if __name__ == "__main__":
    main()
