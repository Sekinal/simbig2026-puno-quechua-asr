"""Figure: same information, different plasticity.

(a) Best-layer linear-probe accuracy of each pretrained omniASR encoder on three stop contrasts
    (trained on dev clips, tested on itest speakers), with the shuffled-label control at the
    same layer. Data: reports/probe_acc_300m.csv, probe_acc_1b.csv, probe_acc.csv (7B).
(b) Linear CKA between the pretrained and the adapted encoder (dev frames) against relative
    depth. 300M and 1B: encoder-only LoRA; 7B: the enc+dec adapter (the only one probed).
    Data: reports/probe_cka_300m.csv, probe_cka_1b.csv, probe_cka.csv.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import style as s

MODELS = [  # name, acc csv, cka csv, number of layers, colour, label
    ("300M", "probe_acc_300m.csv", "probe_cka_300m.csv", 24, s.ENC_RAMP[0], "300M"),
    ("1B", "probe_acc_1b.csv", "probe_cka_1b.csv", 48, s.ENC_RAMP[1], "1B"),
    ("7B", "probe_acc.csv", "probe_cka.csv", 128, s.ENC_RAMP[2], "7B"),
]
TASKS = [
    ("ejective vs plain", "ejective vs plain"),
    ("aspirated vs plain", "aspirated vs plain"),
    ("q vs k (plain)", "/q/ vs /k/"),
]
DODGE = [-0.22, 0.0, 0.22]
KEY_Y = 0.935  # figure-fraction height of the key line above panel a


def main() -> None:
    s.setup()
    fig = plt.figure(figsize=(s.TEXTWIDTH, 3.6 * s.CM))
    ax_a = fig.add_axes((0.235, 0.235, 0.25, 0.61))
    ax_b = fig.add_axes((0.635, 0.235, 0.215, 0.61))

    # ---- (a) probes ---------------------------------------------------------------------------
    s.xgrid(ax_a)
    for (_name, acc_csv, _, _, colour, _), dy in zip(MODELS, DODGE, strict=True):
        acc = s.read(acc_csv)
        for row, (task, _) in enumerate(TASKS):
            best = (
                acc.filter(pl.col("task") == task)
                .sort("acc_zs", descending=True)
                .row(0, named=True)
            )
            y = row + dy
            ax_a.scatter(
                [100 * best["acc_zs"]], [y], s=s.DOT_SMALL, color=colour, linewidths=0, zorder=4
            )
            ax_a.scatter([100 * best["control_zs"]], [y], s=9, color=s.REF, linewidths=0, zorder=3)
    for row, (_, label) in enumerate(TASKS):
        ax_a.text(
            -0.03,
            row,
            label,
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=s.FS,
            color=s.INK,
        )
    # one inline key above panel a (shades = encoder size, as in panel b's end labels)
    tr = fig.transFigure
    for x, colour, text, size in (
        (0.15, s.ENC_RAMP[0], "300M", s.DOT_SMALL),
        (0.234, s.ENC_RAMP[1], "1B", s.DOT_SMALL),
        (0.294, s.ENC_RAMP[2], "7B", s.DOT_SMALL),
        (0.356, s.REF, "shuffled labels", 9),
    ):
        ax_a.scatter([x], [KEY_Y], s=size, color=colour, linewidths=0, transform=tr, clip_on=False)
        ax_a.text(
            x + 0.011,
            KEY_Y,
            text,
            transform=tr,
            ha="left",
            va="center",
            fontsize=s.FS_SMALL,
            color=s.INK,
        )
    ax_a.set_xlim(43, 100)
    ax_a.set_xticks([50, 60, 70, 80, 90, 100])
    ax_a.set_ylim(2.6, -0.45)
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlabel(r"probe accuracy, no adaptation (\%)")

    # ---- (b) CKA vs depth ---------------------------------------------------------------------
    s.ygrid(ax_b)
    for z, (name, _, cka_csv, n_layers, colour, label) in zip((5, 4, 3), MODELS, strict=True):
        cka = s.read(cka_csv).filter(pl.col("set") == "dev").sort("layer")
        x = (cka["layer"] / (n_layers - 1)).to_list()
        y = cka["cka_zs_ed"].to_list()
        ax_b.plot(x, y, color=colour, lw=1.2, zorder=z)
        ax_b.scatter(x, y, s=4, color=colour, linewidths=0, zorder=z)
        lab = label if name != "7B" else "7B (enc+dec)"
        ax_b.annotate(
            lab,
            (x[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=s.FS_SMALL,
            color=s.INK,
            annotation_clip=False,
        )
    ax_b.set_xlim(0, 1)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
    ax_b.set_ylim(0.4, 1.02)
    ax_b.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax_b.set_xlabel(r"relative depth, input $\rightarrow$ output")
    ax_b.set_ylabel("CKA to pretrained")

    fig.text(0.0, 0.93, r"\textbf{a}", fontsize=s.FS, ha="left", va="bottom")
    fig.text(0.545, 0.93, r"\textbf{b}", fontsize=s.FS, ha="left", va="bottom")
    s.save(fig, "plasticity")


if __name__ == "__main__":
    main()
