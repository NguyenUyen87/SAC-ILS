"""Figures of the SAC-ILS manuscript, from the result files (no hand-copied numbers).

Artwork follows the Journal of Heuristics instructions: Arial lettering of uniform size
(8 pt), lines of at least 0.3 pt, information carried by line style and marker as well as
colour, lowercase panel letters, no titles inside the figure. Each figure is drawn at the
width of the text block (372 pt) and included without scaling; its two panels have the same
size and are aligned, and one shared legend sits below them.

Figure chains: (a) mean number of singles by instance size for SAC-ILS and the TBLS-based
controls; (b) per-instance comparison of SAC-ILS and ILS-ADJ.
Figure exact: eight threads; (a) mean number of singles at the end of the budget by
instance size; (b) mean number of singles over the budget at n = 10,000.

    python analysis/make_figures.py [--results results/local] [--out DIR]
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import make_numbers  # noqa: E402
from make_numbers import runs  # noqa: E402

OUT = None
WIDTH = 372 / 72.27                                   # \textwidth of the sn-jnl template, in inches
HEIGHT = 2.55
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.0,
    "lines.markersize": 4, "axes.grid": True, "grid.color": "#d0d0d0", "grid.linewidth": 0.4,
    "grid.linestyle": "-", "axes.axisbelow": True, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
# colour (Okabe-Ito), marker, line style; filled marker only for SAC-ILS
STYLE = {"SACILS": ("SAC-ILS", "#D55E00", "o", "-"), "ILS[adj]": ("ILS-ADJ", "#0072B2", "s", "--"),
         "TBLS+ILS": ("TBLS-ILS", "#009E73", "^", "-."), "TBLS": ("TBLS", "#000000", "v", ":"),
         "ILSx8": ("SAC-ILS (best of 8)", "#D55E00", "o", "-"),
         "CP-SAT": ("CP-SAT", "#0072B2", "s", "--"),
         "CPSAT-M1pre": ("CP-SAT with preprocessing", "#CC79A7", "D", "-."), "HYB": ("HYB", "#E69F00", "x", (0, (1, 1)))}


def layout(top, height):
    """Two equal, aligned panels; the legend goes above them (top = fraction of the height
    reserved for it) and the panel labels (a), (b) are centred below each panel."""
    fig = plt.figure(figsize=(WIDTH, height))
    bottom_in = 0.62                                   # tick labels, axis label and panel label, inches
    left, gap, right = 0.115, 0.125, 0.035
    bottom = bottom_in / height
    w = (1 - left - gap - right) / 2
    h = 1 - bottom - top
    axes = [fig.add_axes([left, bottom, w, h]), fig.add_axes([left + w + gap, bottom, w, h])]
    for ax, s in zip(axes, ("(a)", "(b)")):
        x0 = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x0, 0.03 / height, s, ha="center", va="bottom")
    return fig, axes


def ticks(axis, values, labels=None):
    axis.set_major_locator(FixedLocator(values))
    axis.set_minor_locator(NullLocator())
    fmt = (lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}") if labels is None else \
          (lambda v, _: labels[list(values).index(v)] if v in values else "")
    axis.set_major_formatter(FuncFormatter(fmt))


def plot(ax, x, y, key):
    lab, c, mk, ls = STYLE[key]
    ax.plot(x, y, color=c, marker=mk, ls=ls, mfc=c if key in ("SACILS", "ILSx8") else "white", mew=0.9)


def handle(key):
    lab, c, mk, ls = STYLE[key]
    return Line2D([], [], color=c, marker=mk, ls=ls, mfc=c if key in ("SACILS", "ILSx8") else "white", mew=0.9,
                  label=lab)


def fig_chains():
    sets = [("large", runs("large")), ("scal", runs("scal"))]
    rows = []
    for _, d in sets:
        for n, g in d.groupby("n"):
            for m in ("SACILS", "ILS[adj]", "TBLS+ILS", "TBLS"):
                h = g[g.method == m]
                if not h.empty:
                    rows.append((int(n), m, h.groupby("name").singles.mean().mean()))
    r = pd.DataFrame(rows, columns=["n", "m", "mean"])
    fig, (a, b) = layout(top=0.17, height=2.75)
    order = ("SACILS", "ILS[adj]", "TBLS+ILS", "TBLS")
    for m in order:
        g = r[r.m == m].sort_values("n")
        plot(a, g.n, g["mean"], m)
    a.set_xscale("log"); a.set_yscale("log")
    ticks(a.xaxis, [1000, 2000, 5000, 10000])
    ticks(a.yaxis, [1, 3, 10, 30, 100, 300])
    a.set_xlim(800, 12500); a.set_ylim(1, 300)
    a.set_xlabel("Agents per side $n$"); a.set_ylabel("Mean number of singles")
    for (name, d), mk in zip(sets, ("o", "s")):
        p = d.groupby(["name", "method"]).singles.mean().unstack()
        b.scatter(p["SACILS"] + 1, p["ILS[adj]"] + 1, s=12, marker=mk, facecolors="none",
                  edgecolors="#D55E00" if name == "large" else "#0072B2", linewidths=0.7, clip_on=False, zorder=3)
    b.plot([0.8, 300], [0.8, 300], color="#555555", lw=0.6, ls="--")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(0.8, 300); b.set_ylim(0.8, 300)
    ticks(b.xaxis, [1, 3, 10, 30, 100, 300]); ticks(b.yaxis, [1, 3, 10, 30, 100, 300])
    b.set_xlabel("SAC-ILS: singles + 1"); b.set_ylabel("ILS-ADJ: singles + 1")
    hs = [handle(m) for m in order] + [
        Line2D([], [], marker="o", ls="", mfc="none", mec="#D55E00", label="large set (b)"),
        Line2D([], [], marker="s", ls="", mfc="none", mec="#0072B2", label="scalability set (b)")]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.4, handlelength=2.4, borderaxespad=0.1)
    fig.savefig(os.path.join(OUT, "chains.pdf"))
    plt.close(fig)


def eight_threads():
    frames = []
    for kind in ("large", "scal", "huge"):
        for pre in ("eight_threads", "delorme"):
            f = os.path.join(make_numbers.RUNS, f"{pre}_{kind}.csv")
            if os.path.exists(f) and os.path.getsize(f) > 0:
                x = pd.read_csv(f)
                if not x.empty:
                    frames.append(x)
    x = pd.concat(frames, ignore_index=True)
    assert (x.stable == 1).all()
    for c in ("10", "25", "50", "100"):
        x[f"s{c}"] = np.where(x[f"size_{c}"] >= 0, x.n - x[f"size_{c}"], np.nan)
    return x


def fig_exact():
    x = eight_threads()
    fig, (a, b) = layout(top=0.1, height=2.5)
    order = ("ILSx8", "CP-SAT", "CPSAT-M1pre", "HYB")
    sizes = sorted(x[x.method == "ILSx8"].n.unique())
    for m in order:
        g = x[(x.method == m) & x.n.isin(sizes)].groupby("n").s100.mean()
        plot(a, g.index, g.values, m)
    a.set_xscale("log"); a.set_yscale("log")
    ticks(a.xaxis, [1000, 10000, 100000], ["1,000", "10,000", "100,000"])
    ticks(a.yaxis, [1, 10, 100, 1000])
    a.set_xlim(800, 125000); a.set_ylim(1, 2000)
    a.set_xlabel("Agents per side $n$"); a.set_ylabel("Mean number of singles")
    g = x[x.n == 10000]
    fr = [0.1, 0.25, 0.5, 1.0]
    for m in order:
        h = g[g.method == m]
        plot(b, fr, [h[f"s{int(100 * f)}"].mean() for f in fr], m)
    b.set_xscale("log"); b.set_yscale("log")
    ticks(b.xaxis, fr, ["10%", "25%", "50%", "100%"])
    ticks(b.yaxis, [10, 20, 50, 100])
    b.set_xlim(0.085, 1.15); b.set_ylim(10, 130)
    b.set_xlabel("Fraction of the budget ($n$ = 10,000)"); b.set_ylabel("Mean number of singles")
    fig.legend(handles=[handle(m) for m in order], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.4, borderaxespad=0.1)
    fig.savefig(os.path.join(OUT, "exact.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="result directory (default: results/reference)")
    ap.add_argument("--out", default=None, help="output directory (default: analysis/out/<name of the result directory>)")
    args = ap.parse_args()
    make_numbers.configure(args.results, args.out)
    OUT = make_numbers.OUT
    os.makedirs(OUT, exist_ok=True)
    fig_chains()
    fig_exact()
    print("figures written to", OUT)
