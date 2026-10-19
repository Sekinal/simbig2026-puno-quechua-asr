"""Shared style for every paper figure (imported by each ``fig_<name>.py``).

One palette with a fixed meaning across all figures:
  decoder adaptation = blue, the encoder and its adaptation = orange (shades of orange =
  encoder size, light -> dark), zero-shot / reference = grey,
  synthetic verifiers = aqua, the sentence-memorizing word 4-gram system = violet,
  our system = ink.
Text is always set in ink colours, never in series colours. Figures are drawn at their final
printed size (LNCS text width 12.2 cm) so 7-8 pt here is 7-8 pt on the page; text is typeset
by LaTeX (Computer Modern) to match the body of the paper.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIG_DATA = REPORTS / "fig_data"
OUT = ROOT / "docs" / "paper" / "figs"

CM = 1 / 2.54
TEXTWIDTH = 12.2 * CM  # LNCS \textwidth in inches
HALFWIDTH = 5.9 * CM  # one of two side-by-side figures (with a small gap)

# Palette (validated colour-blind safe as a set: blue / orange / aqua / violet).
DEC = "#2a78d6"  # decoder adaptation
ENC = "#e8602c"  # encoder adaptation
ENC_RAMP = (
    "#eb8a52",
    "#e8602c",
    "#a63d12",
)  # encoder size 300M -> 1B -> 7B (one hue, light -> dark)
REF = "#a8a7a2"  # zero-shot / reference
SYN = "#169c6c"  # synthetic verifiers
LM4 = "#5a48b5"  # CTC + word 4-gram on the training prompts
INK = "#1d1d1b"  # primary text and our system
INK2 = "#55544f"  # secondary text, axes
MUTED = "#74736e"  # tertiary text (>= 7 pt only)
GRID = "#e9e8e4"  # recessive grid
SHADE = "#d9d8d3"  # unshaded schematic cell outline / fill

FS = 8  # base font size (pt): row and category labels
FS_SMALL = 7  # minimum text size
DOT = 28  # marker area (pt^2) of a main data point, in every figure
DOT_SMALL = 16  # marker area of a secondary data point


def setup() -> None:
    """Apply the shared rcParams."""
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
            "font.family": "serif",
            "font.size": FS,
            "axes.labelsize": FS,
            "axes.labelcolor": INK2,
            "axes.titlesize": FS,
            "axes.edgecolor": INK2,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": FS_SMALL,
            "ytick.labelsize": FS_SMALL,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "text.color": INK,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
            "figure.facecolor": "white",
        }
    )


def ygrid(ax) -> None:
    """Light, recessive horizontal grid behind the data."""
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def xgrid(ax) -> None:
    """Light, recessive vertical grid behind the data."""
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def read(name: str, fig_data: bool = False) -> pl.DataFrame:
    """Read a CSV from ``reports/`` (or ``reports/fig_data/``)."""
    return pl.read_csv((FIG_DATA if fig_data else REPORTS) / name)


def save(fig, name: str) -> Path:
    """Write ``docs/paper/figs/<name>.pdf`` at the drawn size (no tight-bbox resizing)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "Producer": None})
    plt.close(fig)
    return path


def fmt(x: float, nd: int = 1) -> str:
    """Round half away from zero on the decimal value as written in the CSV (6.35 -> 6.4)."""
    q = Decimal(1).scaleb(-nd)
    return str(abs(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)))


def signed(x: float, nd: int = 1) -> str:
    """Format a difference with a true minus sign (LaTeX math)."""
    v = fmt(x, nd)
    if float(v) == 0:
        return v
    return f"$-${v}" if x < 0 else f"$+${v}"


def sig_colour(lo: float, hi: float) -> str:
    """Ink for an interval that excludes 0, muted grey for one that includes it (n.s.)."""
    return INK if lo * hi > 0 else MUTED


def ci(lo: float, hi: float, nd: int = 1) -> str:
    """Format a 95% interval as [lo, hi] with signs."""
    return f"[{signed(lo, nd)}, {signed(hi, nd)}]"
