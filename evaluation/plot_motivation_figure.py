"""Figure for the Motivation section, and the exploration behind it.

Standard STL decomposition (observed, trend, seasonal, residual) of the
daily active cases of a log over its training period, with two moments t1
and t2 marked through all rows. The decomposition uses the whole series
(also the days after t1 and t2), so it only illustrates the problem; the
approach itself uses an expanding-window decomposition.

--explore: one figure per log in --dataset (default: all), and a table with
the strength of trend and seasonal (Hyndman & Athanasopoulos:
F = max(0, 1 - var(R) / var(X + R))) and the variance share of each
component; with --top, candidate pairs of moments per log.
Output: evaluation/results/motivation/decompositions/<ds>.png and strength.csv

--explore --paper --days: the figure of the paper. Observed series in blue;
t1 in red (solid line, circle) and t2 in green (dashed line, triangle), so
that the moments also differ in grayscale print. LNCS text width.
Output: evaluation/results/motivation/figure_motivation.pdf and .png

Usage:
    python evaluation/plot_motivation_figure.py --explore
    python evaluation/plot_motivation_figure.py --explore --dataset bpic20155 --top 15
    python evaluation/plot_motivation_figure.py --explore --dataset bpic20155 --days 2011-04-09 2011-06-11
    python evaluation/plot_motivation_figure.py --explore --paper --dataset bpic20155 --days 2011-02-09 2012-07-18
"""
import glob
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")              # no display needed (server)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))   # evaluation/: scripts and results/
ROOT = os.path.dirname(EVAL_DIR)                         # repository root: data/
PREP_DIR = os.path.join(ROOT, "data", "prepared")
OUT_DIR = os.path.join(EVAL_DIR, "results", "motivation")
SERIES = "active_cases"
EXCLUDED_LOGS = ["credit", "invoice"]

CM = 1 / 2.54
WIDTH = 12.2 * CM                      # LNCS text width
COLORS = ("#0072B2", "#D55E00")        # Okabe-Ito blue, vermillion
MARKERS = ("o", "^")
# --paper: the observed series in blue; the two moments in red (solid line,
# circle) and green (dashed line, triangle). Red and green are close for
# red-green colorblind readers and in grayscale print, so the line style and
# the marker shape keep the moments apart (secondary encoding).
PAPER_LINE = "#0072B2"
PAPER_COLORS = ("#C62828", "#2E9E44")
LINESTYLES = ("-", "--")
SERIES_GRAY, TREND_GRAY, GRID_GRAY = "#A6A6A6", "#2B2B2B", "#E6E6E6"


def set_paper_style():
    """Serif fonts at caption size, thin recessive axes."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def load_active_cases(ds, start):
    """Daily active cases of the training period (before the first test
    case), from start on; by default from the first day with at least 10%
    of the maximum (leaves out the start of the log, where the cases that
    were already running are not recorded)."""
    with open(os.path.join(PREP_DIR, f"{ds}_series.pkl"), "rb") as f:
        s = pickle.load(f)
    series = {ser.name: ser for ser in s["series"]}[SERIES]
    series = series[series.index < s["test_start"]].astype(float)
    if start is None:
        start = series.index[np.argmax(series.values >= 0.1 * series.max())]
    return series[series.index >= pd.Timestamp(start)]


def list_datasets():
    """All logs with a prepared series file, without the excluded ones."""
    paths = glob.glob(os.path.join(PREP_DIR, "*_series.pkl"))
    names = sorted(os.path.basename(p)[:-len("_series.pkl")] for p in paths)
    return [n for n in names if n not in EXCLUDED_LOGS]


def decompose_series(series, period, trend_window):
    """Standard STL: trend, seasonal and residual as series on the same index."""
    from statsmodels.tsa.seasonal import STL
    fit = STL(series.values, period=period, trend=trend_window | 1, robust=True).fit()
    return {"trend": pd.Series(fit.trend, index=series.index),
            "seasonal": pd.Series(fit.seasonal, index=series.index),
            "residual": pd.Series(fit.resid, index=series.index)}


def compute_strength(series, comp):
    """Strength of trend and seasonal, and the variance share of each component."""
    t, s, r = comp["trend"], comp["seasonal"], comp["residual"]
    total = series.var()
    return {"days": len(series),
            "trend_strength": max(0.0, 1 - r.var() / (t + r).var()),
            "seasonal_strength": max(0.0, 1 - r.var() / (s + r).var()),
            "share_trend": t.var() / total, "share_seasonal": s.var() / total,
            "share_residual": r.var() / total}


def plot_decomposition(ds, series, comp, strength, path, days=(), history_days=7, paper=False):
    """4 rows on a shared time axis: series, trend, seasonal, residual. With
    days: each moment is marked through all rows (history window, a vertical
    line, its marker, and the value of each component). paper: the version
    for the paper (no title and no history windows, readable value labels,
    also saved as PDF)."""
    if paper:    # larger text for the paper
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                             "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, (11 if paper else 9) * CM), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 0.8, 0.8] if not paper
                                          else [1, 1, 1, 1], "hspace": 0.25})
    if paper:    # room on the right for the values at t1 and t2
        fig.subplots_adjust(left=0.1, right=0.88)
    # paper: the standard STL names (observed = trend + seasonal + residual),
    # all in number of active cases (said in the caption)
    first = "observed" if paper else "series"
    line = TREND_GRAY    # trend, seasonal and residual stay dark gray; only observed is blue
    moment_colors = PAPER_COLORS if paper else COLORS
    rows = [(first, series, PAPER_LINE if paper else SERIES_GRAY, 0.7),
            ("trend", comp["trend"], line, 1.3),
            ("seasonal", comp["seasonal"], line, 0.6),
            ("residual", comp["residual"], line, 0.6)]
    for ax, (label, values, color, lw) in zip(axes, rows):
        ax.plot(values.index, values.values, color=color, lw=lw)
        if label in ("seasonal", "residual"):
            ax.axhline(0, color=TREND_GRAY, lw=0.5)
        if paper:
            ax.set_ylabel(label)
        else:
            ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.grid(axis="y", color=GRID_GRAY, lw=0.5)
        ax.set_axisbelow(True)
        for k, (day, color, marker) in enumerate(zip(days, moment_colors, MARKERS), 1):
            if not paper:
                ax.axvspan(day - pd.Timedelta(days=history_days), day, color=color, alpha=0.18, lw=0)
            ax.axvline(day, color=color, lw=0.8 if paper else 0.6, zorder=2,
                       ls=LINESTYLES[k - 1] if paper else "-")
            ax.plot(day, values[day], marker=marker, ms=4.5, color=color,
                    markeredgecolor="white", markeredgewidth=0.6, ls="none", zorder=5)
            text = f"{values[day]:+.1f}" if label in ("seasonal", "residual") else f"{values[day]:.0f}"
            if paper:    # values in the right margin, one line per moment, never on the curves
                ax.text(1.01, 0.68 - 0.36 * (k - 1), f"$t_{k}$: {text}", transform=ax.transAxes,
                        ha="left", va="center", fontsize=8.5, color=color)
            else:
                ax.annotate(text, (day, values[day]), xytext=(4, 3), textcoords="offset points",
                            fontsize=6.5, color=color)
            if label == first:
                ax.annotate(f"$t_{k}$", (day, ax.get_ylim()[1]), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=10 if paper else 8, color=color if paper else "black")
    if paper:    # month and year, every half year
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{mdates.num2date(x).month}.{mdates.num2date(x).year}"))
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if not paper:
        axes[0].set_title(f"{ds}   ($F_T$ = {strength['trend_strength']:.2f}, "
                          f"$F_S$ = {strength['seasonal_strength']:.2f})", loc="left",
                          pad=12 if len(days) else 3)
    fig.align_ylabels(axes)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    if paper:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def rank_day_pairs(series, comp, history_days, max_state_diff, top):
    """Pairs of days on the same weekday with (almost) the same value, ranked by
    how differently that value is made up: the difference in trend (the same
    value then comes from a high level plus a negative residual on one day,
    and from a low level plus a positive residual on the other)."""
    days = series.index[series.index >= series.index[0] + pd.Timedelta(days=history_days)]
    rows = []
    for i, d1 in enumerate(days):
        for d2 in days[i + 1:]:
            if d1.dayofweek != d2.dayofweek or abs(series[d1] - series[d2]) > max_state_diff:
                continue
            rows.append({"day_1": d1.date(), "day_2": d2.date(), "weekday": f"{d1:%a}",
                         "value_1": series[d1], "value_2": series[d2],
                         **{f"{c}_{k}": comp[c][d] for c in ("trend", "seasonal", "residual")
                            for k, d in ((1, d1), (2, d2))},
                         "trend_diff": abs(comp["trend"][d1] - comp["trend"][d2])})
    table = pd.DataFrame(rows).sort_values("trend_diff", ascending=False).head(top)
    cols = ["day_1", "day_2", "weekday", "value_1", "value_2", "trend_1", "trend_2",
            "seasonal_1", "seasonal_2", "residual_1", "residual_2", "trend_diff"]
    return table[cols]


def explore_datasets(datasets, period, trend_window, days=(), history_days=7,
                     max_state_diff=2, top=0, paper=False):
    """Decomposition figure per log and one strength table for all of them.
    With days, the moments are marked; with top > 0, candidate pairs of
    moments are listed (and saved) per log."""
    out_dir = os.path.join(OUT_DIR, "decompositions")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ds in datasets:
        try:
            series = load_active_cases(ds, None)
        except (KeyError, FileNotFoundError) as e:
            print(f"skipped {ds}: {e!r}")
            continue
        comp = decompose_series(series, period, trend_window)
        strength = compute_strength(series, comp)
        for d in days:
            if d not in series.index:
                raise SystemExit(f"{ds}: {d.date()} is not in the shown training period")
        suffix = "_" + "_".join(f"{d:%Y%m%d}" for d in days) if days else ""
        path = (os.path.join(OUT_DIR, "figure_motivation.png") if paper
                else os.path.join(out_dir, f"{ds}{suffix}.png"))
        plot_decomposition(ds, series, comp, strength, path, days, history_days, paper)
        if paper:
            print("written:", path, "and .pdf")
        rows.append({"dataset": ds, **strength})
        for k, d in enumerate(days, 1):
            print(f"{ds} t{k} = {d.date()} ({d:%a}): series {series[d]:.0f} = trend "
                  f"{comp['trend'][d]:.1f} + seasonal {comp['seasonal'][d]:+.1f} "
                  f"+ residual {comp['residual'][d]:+.1f}")
        if top > 0:
            pairs = rank_day_pairs(series, comp, history_days, max_state_diff, top)
            pairs.to_csv(os.path.join(out_dir, f"{ds}_pairs.csv"), index=False)
            with pd.option_context("display.width", 250, "display.precision", 1):
                print(f"\n{ds}: candidate pairs\n" + pairs.to_string(index=False) + "\n")
    table = pd.DataFrame(rows).sort_values("seasonal_strength", ascending=False)
    table.to_csv(os.path.join(out_dir, "strength.csv"), index=False)
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(table.to_string(index=False))
    print("written:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="logs to decompose (default: all); with --paper the log of the figure")
    parser.add_argument("--days", nargs=2, default=None, help="the two moments, YYYY-MM-DD")
    parser.add_argument("--explore", action="store_true",
                        help="decompose several logs to choose the log and the moments")
    parser.add_argument("--paper", action="store_true",
                        help="with --explore and --days: the figure for the paper, "
                             "results/motivation/figure_motivation.pdf")
    parser.add_argument("--top", type=int, default=0,
                        help="with --explore: list this many candidate pairs of moments per log")
    parser.add_argument("--max-state-diff", type=float, default=2,
                        help="with --top: largest difference in active cases between the two moments")
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--period", type=int, default=7, help="seasonal period of the STL for the trend")
    parser.add_argument("--trend-window", type=int, default=61,
                        help="trend smoothing window of the STL in days (odd); larger = smoother")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_paper_style()
    if not args.explore:
        raise SystemExit("use --explore; with --paper and --days it writes the figure of the paper")
    if args.paper and not args.days:
        raise SystemExit("--paper needs --days")
    explore_datasets(args.dataset or list_datasets(), args.period, args.trend_window,
                     [pd.Timestamp(d) for d in args.days or []], args.history_days,
                     args.max_state_diff, args.top, args.paper)


if __name__ == "__main__":
    main()