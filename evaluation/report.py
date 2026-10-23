"""All results of the paper from the trained runs, with one command; trains
nothing.

For every run in --runs (written by experiment_lag.py, evaluation/results/runs/<run>/):
  1. selection on the validation set: for every candidate setting the
     validation MAE is averaged over the seeds, the setting with the lowest
     mean is selected, and the test MAE of that one setting is reported
     (mean and std over the seeds). F3 (raw lags without decomposition)
     needs k >= MIN_K_RAW. Two selections:
       per k   for every k, the seasonal period with the lowest mean
               validation MAE -> one test result per k
       joint   the pair (period, k) with the lowest mean validation MAE
               -> one test result per log (the result reported in the paper)
     The test MAE does not influence the selection, so the result is the
     same as if the test set were evaluated only for the selected setting.
  2. all figures of the run
Once for all runs:
  3. the characteristics of the logs (CV and variance fractions of the
     daily time series on the training period), computed if missing
  4. the table of the test MAE (all runs side by side), as CSV
  5. the daily time series of each log

Output, all in evaluation/results/report/:
    log_characteristics.csv       one row per log and time series
    mae_table.csv                 the table of the paper (one row per log)
    time_series/<log>.png         daily time series, test start dashed
    <run>/selected.csv            joint selection: one row per log and feature set
    <run>/mae_per_log.csv         as selected.csv, with the baseline B, for the table
    <run>/per_k.csv               per-k selection: one row per log, feature set and k
    <run>/wins.csv, wins_per_k.csv  logs at least as good as B, clear wins, sign test
    <run>/comparison.csv          per log and feature set: joint selection vs all k
    <run>/figures/
        mae_per_log(.png/.pdf)          test MAE of B, F3, F4, F4ts per log (main figure)
        mae_per_log_from_zero(.png/.pdf)  as mae_per_log, every x axis from 0
        k_selection_<set>(.png/.pdf)    smallest allowed k vs k selected on validation
        compare_F4_vs_F4ts(.png/.pdf)   F4 vs F4 without residual, with CV and variance fractions
        selected.png                    change vs B per log, joint selection
        per_k.png                       test MAE against k, period selected per k
        per_k_with_selection.png        as per_k, star: joint selection
        wins_per_k.png                  logs at least as good as B, per k
        all_periods.png                 test MAE against k for every period (no selection)
        by_k/k<k>.png                   one figure per k: change vs B for all logs

The figure of the Motivation section is made by plot_motivation_figure.py.

Usage:
    python evaluation/report.py
    python evaluation/report.py --runs xgb_periods rf_periods
    python evaluation/report.py --runs xgb_periods --refresh-characteristics
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")              # no display needed (server)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))   # evaluation/: scripts and results/
ROOT = os.path.dirname(EVAL_DIR)                         # repository root: dyRtp/, configs/, data/
sys.path.insert(0, ROOT)                                 # so that dyRtp is found

from dyRtp.utilities import analyze_series_stats, compute_variance_fractions, select_train_period

RUNS_DIR = os.path.join(EVAL_DIR, "results", "runs")
PREP_DIR = os.path.join(ROOT, "data", "prepared")
REPORT_DIR = os.path.join(EVAL_DIR, "results", "report")
CHARACTERISTICS_FILE = os.path.join(REPORT_DIR, "log_characteristics.csv")
TABLE_FILE = os.path.join(REPORT_DIR, "mae_table.csv")

EXCLUDED_LOGS = ["credit", "invoice"]       # credit: no resource attribute; invoice: left out
EXCLUDED_VARIANTS = ["F1+F2+F3+F4"]         # not part of the paper
MIN_K_RAW = 2      # F3 without F4: at least two daily values (one value is a state, not a change)
MIN_CYCLES = 6     # as in experiment_lag.py: a period is used only if the fit period covers this many cycles
BASELINE, NO_INTERCASE = "F1+F2+B", "F1+F2"

# feature sets of the paper: (variant, short name, color, marker, legend text)
VARIANTS = [("F1+F2+B", "B", "#555555", "s", "B: baseline, snapshot at prediction time"),
            ("F1+F2+F3", "F3", "#0072B2", "o", "F3: observed lags"),
            ("F1+F2+F4", "F4", "#D55E00", "^", "F4: decomposed lags"),
            ("F1+F2+F4ts", "F4ts", "#CC79A7", "D", "F4ts: decomposed lags without residual")]
COLORS = {v: color for v, _, color, _, _ in VARIANTS if v != BASELINE}
BASELINE_COLOR = "#555555"
K_SELECTION_VARIANTS = ["F1+F2+F3", "F1+F2+F4"]
# pairs of feature sets compared per log, each with (period, k) selected on validation
VARIANT_PAIRS = [("F1+F2+F4", "F1+F2+F4ts", "F4", "F4 without residual")]
PERIOD_MARKERS = {0: "o", 7: "o", 14: "s", 30: "^"}
PERIOD_STYLES = {0: "-", 7: "-", 14: "--", 30: ":"}

# order of the logs in the table: by family, then by name
LOG_ORDER = ["bpic2011", "bpic2012a", "bpic2012o", "bpic2012w", "bpic20151", "bpic20152",
             "bpic20153", "bpic20154", "bpic20155", "bpic2017", "helpdesk", "hospital",
             "production", "sepsis", "traffic_fines"]
PAPER_STYLE = {"font.family": "serif", "font.size": 8, "axes.spines.top": False,
               "axes.spines.right": False, "axes.linewidth": 0.6}


def save_figure(fig, path, pdf=False):
    """Save as PNG (and PDF for the figures of the paper) and close."""
    fig.savefig(path, dpi=250 if pdf else 150, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ===================================================== 1. selection on validation

def load_results(run_name):
    """One row per log, feature set, period, k and seed, with test MAE (mae)
    and validation MAE (val_mae)."""
    results = pd.read_csv(os.path.join(RUNS_DIR, run_name, "all_logs.csv"))
    if "period" not in results.columns:
        results["period"] = 0
    if "val_mae" not in results.columns:
        raise ValueError("this run has no validation MAE; selection is not possible")
    results["period"] = results["period"].fillna(0).astype(int)
    results = results.drop_duplicates(["dataset", "variant", "period", "k", "seed"])
    return drop_too_long_periods(results)


def drop_too_long_periods(results):
    """Leave out periods that are too long for the log (fit period shorter
    than MIN_CYCLES cycles), as experiment_lag.py does; results of runs
    started before this rule may contain them."""
    for ds in results.dataset.unique():
        path = os.path.join(PREP_DIR, f"{ds}_fit_days.txt")
        if os.path.exists(path):
            with open(path) as f:
                days = int(f.read())
            results = results[~((results.dataset == ds) & (results.period * MIN_CYCLES > days))]
    return results


def summarize_over_seeds(results):
    """Per log, feature set, period and k: mean and std of the test MAE and
    mean validation MAE over the seeds."""
    return (results.groupby(["dataset", "variant", "period", "k"])
            .agg(mean=("mae", "mean"), std=("mae", "std"), n_seeds=("mae", "size"),
                 val_mean=("val_mae", "mean"))
            .reset_index())


def get_smallest_k(variant):
    """Smallest window allowed for a feature set: MIN_K_RAW for raw lags
    without decomposition (F3 without F4), 1 otherwise."""
    return MIN_K_RAW if "F3" in variant and "F4" not in variant else 1


def select_on_validation(summary, by):
    """Per log and windowed feature set (and the columns in `by`), the
    setting with the lowest mean validation MAE over the seeds, among the
    windows k allowed for the feature set (get_smallest_k)."""
    windowed = summary[(summary.k > 0) & summary["val_mean"].notna()]
    windowed = windowed[windowed.k >= windowed.variant.map(get_smallest_k)]
    keys = ["dataset", "variant"] + by
    return windowed.loc[windowed.groupby(keys)["val_mean"].idxmin()].reset_index(drop=True)


def add_change_vs_references(table, summary):
    """Change of the test MAE vs F1+F2+B (and F1+F2) in %, negative = better."""
    refs = summary[summary.k == 0].pivot_table(index="dataset", columns="variant", values="mean")
    table = table.copy()
    table["vs_B_%"] = (table["mean"] / table.dataset.map(refs[BASELINE]) - 1) * 100
    if NO_INTERCASE in refs:
        table["vs_F1+F2_%"] = (table["mean"] / table.dataset.map(refs[NO_INTERCASE]) - 1) * 100
    return table


def count_wins(table, summary, by):
    """Logs with a test MAE lower than / at most the baseline and lower by
    more than one std of the baseline, with the one-sided sign test p-value
    (H0: better and worse equally likely)."""
    base = summary[(summary.k == 0) & (summary.variant == BASELINE)].set_index("dataset")
    rows = []
    for key, g in table.groupby(by + ["variant"]):
        d = g.dropna(subset=["vs_B_%"])
        better, worse = int((d["vs_B_%"] < 0).sum()), int((d["vs_B_%"] > 0).sum())
        clear = int((d["mean"] < d.dataset.map(base["mean"]) - d.dataset.map(base["std"])).sum())
        p = binomtest(better, better + worse, 0.5, alternative="greater").pvalue if better + worse else np.nan
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by + ["variant"], key), better=better, clear_wins=clear,
                         at_least_as_good=int((d["vs_B_%"] <= 0).sum()), n_logs=len(d),
                         median_vs_B_pct=d["vs_B_%"].median(), sign_test_p=p))
    return pd.DataFrame(rows)


def compare_selections(per_k, joint):
    """Per log and feature set: the joint selection, and how the k values of
    the per-k selection compare with the baseline (mean change over all k,
    share of k at least as good as the baseline, best and worst k)."""
    rows = []
    for (ds, variant), g in per_k.groupby(["dataset", "variant"]):
        j = joint[(joint.dataset == ds) & (joint.variant == variant)]
        row = {"dataset": ds, "variant": variant,
               "per_k_mean_vs_B_%": g["vs_B_%"].mean(),
               "per_k_share_at_least_as_good": (g["vs_B_%"] <= 0).mean(),
               "per_k_best_vs_B_%": g["vs_B_%"].min(), "per_k_worst_vs_B_%": g["vs_B_%"].max()}
        if len(j):
            j = j.iloc[0]
            row.update({"joint_period": int(j["period"]), "joint_k": int(j["k"]),
                        "joint_mean": j["mean"], "joint_std": j["std"], "joint_vs_B_%": j["vs_B_%"]})
        rows.append(row)
    return pd.DataFrame(rows)


def build_mae_table(joint, summary):
    """One row per log and feature set of the paper: test MAE (mean, std
    over the seeds), selected period and k, and change vs F1+F2+B in %."""
    base = summary[(summary.k == 0) & (summary.variant == BASELINE)].copy()
    base["vs_B_%"] = 0.0
    sel = joint[joint.variant.isin(COLORS)]
    cols = ["dataset", "variant", "period", "k", "mean", "std", "vs_B_%"]
    table = pd.concat([base[cols], sel[cols]], ignore_index=True)
    return table[table.dataset.isin(sel.dataset)].reset_index(drop=True)


def run_selection(run, exclude, out_dir):
    """Selections of one run, saved as CSV; returns summary, per_k, joint."""
    results = load_results(run)
    results = results[~results.dataset.isin(EXCLUDED_LOGS + exclude)
                      & ~results.variant.isin(EXCLUDED_VARIANTS)]
    summary = summarize_over_seeds(results)
    expected = summary.n_seeds.max()
    short = summary[summary.n_seeds < expected]
    if len(short):
        print(f"  warning: {len(short)} settings have fewer than {expected} seeds "
              f"(e.g. {short.iloc[0][['dataset', 'variant', 'period', 'k']].to_dict()})")
    per_k = add_change_vs_references(select_on_validation(summary, by=["k"]), summary)
    joint = add_change_vs_references(select_on_validation(summary, by=[]), summary)
    per_k.to_csv(os.path.join(out_dir, "per_k.csv"), index=False)
    joint.to_csv(os.path.join(out_dir, "selected.csv"), index=False)
    count_wins(per_k, summary, by=["k"]).to_csv(os.path.join(out_dir, "wins_per_k.csv"), index=False)
    count_wins(joint, summary, by=[]).to_csv(os.path.join(out_dir, "wins.csv"), index=False)
    compare_selections(per_k, joint).to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    build_mae_table(joint, summary).round(4).to_csv(os.path.join(out_dir, "mae_per_log.csv"),
                                                     index=False)
    return summary, per_k, joint


# ===================================================== 2. figures of one run

def get_model_name(run):
    """Name of the learner of a run, from run_info.yaml (XGBoost if not given)."""
    path = os.path.join(RUNS_DIR, run, "run_info.yaml")
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            name = (yaml.safe_load(f).get("model") or {}).get("name", "xgb")
        return {"rf": "Random forest", "lstm": "LSTM"}.get(name, "XGBoost")
    return "XGBoost"


def draw_baseline(ax, summary, ds):
    """The baseline of one log as a dashed line with a band of +- one std."""
    b = summary[(summary.dataset == ds) & (summary.variant == BASELINE) & (summary.k == 0)]
    if len(b):
        m, s = b["mean"].iloc[0], b["std"].iloc[0]
        ax.axhline(m, color=BASELINE_COLOR, ls="--", lw=1, label=BASELINE)
        ax.axhspan(m - s, m + s, color=BASELINE_COLOR, alpha=0.12)


def plot_all_periods(summary, path, ncols=3):
    """One panel per log: test MAE (mean over the seeds, std as error bar)
    against k for every feature set and seasonal period, without selection;
    the star marks the (period, k) selected on validation."""
    logs = sorted(summary.dataset.unique())
    nrows = int(np.ceil(len(logs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, ds in zip(axes.flat, logs):
        g = summary[summary.dataset == ds]
        draw_baseline(ax, summary, ds)
        for variant, color in COLORS.items():
            rv = g[g.variant == variant]
            if rv.empty:
                continue
            periods = sorted(rv["period"].unique())
            for period in periods:
                r = rv[rv.period == period].sort_values("k")
                label = f"{variant} p={period}" if len(periods) > 1 else variant
                ax.errorbar(r["k"], r["mean"], yerr=r["std"], color=color, marker="o", ms=3,
                            capsize=2, lw=1, ls=PERIOD_STYLES.get(period, "-"), label=label)
            allowed = rv[(rv.k >= get_smallest_k(variant)) & rv["val_mean"].notna()]
            if len(allowed):
                best = allowed.loc[allowed["val_mean"].idxmin()]
                ax.plot(best["k"], best["mean"], marker="*", ms=14, color=color,
                        markeredgecolor="black", ls="none")
        ax.set_xticks(sorted(g.loc[g.k > 0, "k"].unique()))
        ax.set_title(ds)
        ax.set_xlabel("k (days)")
        ax.set_ylabel("MAE (days)")
    for ax in list(axes.flat)[len(logs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    save_figure(fig, path)


def plot_per_k(per_k, summary, path, joint=None, ncols=3, title=None):
    """One panel per log: test MAE (mean over the seeds, std as error bar)
    against k, with the period selected on validation for each k (marker:
    circle 7, square 14, triangle 30). If `joint` is given, a star marks the
    (period, k) selected jointly."""
    logs = sorted(summary.dataset.unique())
    nrows = int(np.ceil(len(logs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, ds in zip(axes.flat, logs):
        draw_baseline(ax, summary, ds)
        for variant, color in COLORS.items():
            r = per_k[(per_k.dataset == ds) & (per_k.variant == variant)].sort_values("k")
            if r.empty:
                continue
            ax.errorbar(r["k"], r["mean"], yerr=r["std"], color=color, ms=0, capsize=3, lw=1,
                        label=variant)
            for period, marker in PERIOD_MARKERS.items():
                p = r[r.period == period]
                ax.plot(p["k"], p["mean"], marker=marker, ms=5, color=color, ls="none")
            if joint is not None:
                j = joint[(joint.dataset == ds) & (joint.variant == variant)]
                if len(j):
                    ax.plot(j["k"], j["mean"], marker="*", ms=14, color=color,
                            markeredgecolor="black", ls="none")
        ax.set_xticks(sorted(per_k.k.unique()))
        ax.set_title(ds)
        ax.set_xlabel("k (days)")
        ax.set_ylabel("MAE (days)")
    for ax in list(axes.flat)[len(logs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for period, marker in PERIOD_MARKERS.items():
        if period > 0 and (per_k.period == period).any():
            handles.append(plt.Line2D([], [], color="black", marker=marker, ls="none", ms=5))
            labels.append(f"period {period}")
    if joint is not None:
        handles.append(plt.Line2D([], [], color="grey", marker="*", ms=12, markeredgecolor="black",
                                  ls="none"))
        labels.append("(period, k) selected jointly")
    axes.flat[0].legend(handles, labels, fontsize=7)
    fig.suptitle(title or "Period selected on validation for each k (mean over the seeds)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def plot_wins_per_k(wins, path):
    """Grouped bars: per k and feature set, logs at least as good as the
    baseline (light) and better by more than one std of the baseline (dark)."""
    ks = sorted(wins.k.unique())
    variants = [v for v in COLORS if v in set(wins.variant)]
    width = 0.8 / len(variants)
    n_logs = int(wins.n_logs.max())
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(ks) + 2) + 3, 4.2))
    for i, variant in enumerate(variants):
        c = wins[wins.variant == variant].set_index("k").reindex(ks)
        x = np.arange(len(ks)) + (i - (len(variants) - 1) / 2) * width
        ax.bar(x, c["at_least_as_good"], width, color=COLORS[variant], alpha=0.45,
               edgecolor=COLORS[variant], label=f"{variant}: MAE <= B")
        ax.bar(x, c["clear_wins"], width, color=COLORS[variant], label=f"{variant}: better by > 1 std")
    ax.axhline(n_logs / 2, color="black", ls=":", lw=1, label=f"half of the logs ({n_logs / 2:g})")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_xlabel("k (days)")
    ax.set_ylabel("number of logs")
    ax.set_ylim(0, n_logs + 0.5)
    ax.set_title("Logs at least as good as F1+F2+B (period selected on validation for each k)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save_figure(fig, path)


def plot_selected(joint, summary, path):
    """Grouped bars per log: change of the test MAE vs F1+F2+B in % for each
    feature set with its (period, k) selected jointly on validation (written
    on the bar); error bar: std over the seeds in % of the baseline."""
    base = summary[(summary.k == 0) & (summary.variant == BASELINE)].set_index("dataset")["mean"]
    logs = sorted(ds for ds in summary.dataset.unique() if ds in base.index)
    variants = [v for v in COLORS if v in set(joint.variant)]
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(logs)) + 3, 5))
    for i, variant in enumerate(variants):
        g = joint[joint.variant == variant].set_index("dataset").reindex(logs)
        x = np.arange(len(logs)) + (i - (len(variants) - 1) / 2) * width
        err = (g["std"] / base.reindex(logs) * 100).values
        ax.bar(x, g["vs_B_%"].values, width, color=COLORS[variant], label=variant,
               yerr=err, capsize=2, error_kw={"lw": 0.8})
        for xi, e, (_, r) in zip(x, err, g.iterrows()):
            if np.isnan(r["vs_B_%"]):
                continue
            label = f"k{int(r['k'])}" if r["period"] == 0 else f"p{int(r['period'])} k{int(r['k'])}"
            below = r["vs_B_%"] < 0
            end = r["vs_B_%"] + (-1 if below else 1) * (0 if np.isnan(e) else e)
            ax.annotate(label, (xi, end), rotation=90, fontsize=6, ha="center",
                        va="top" if below else "bottom",
                        xytext=(0, -3 if below else 3), textcoords="offset points")
    ax.margins(y=0.15)
    ax.axhline(0, color=BASELINE_COLOR, ls="--", lw=1.2, label=f"{BASELINE} (baseline)")
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels(logs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE change vs F1+F2+B (%)")
    ax.set_title("Period and k selected jointly on validation (negative = better)", fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save_figure(fig, path)


def plot_vs_baseline_for_k(per_k, summary, k, path, clip=20.0):
    """One figure for one k: for every log (x axis) the change of the test
    MAE relative to the baseline in % (negative = better), for each feature
    set at this k (period selected on validation). Error bars: std over the
    seeds in % of the baseline. Values beyond +-clip % are drawn at the edge
    with their value."""
    base = summary[(summary.k == 0) & (summary.variant == BASELINE)].set_index("dataset")["mean"]
    at_k = per_k[per_k.k == k]
    logs = sorted(ds for ds in summary.dataset.unique() if ds in base.index)
    variants = [v for v in COLORS if v in set(at_k.variant)]
    if not variants:
        return
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(logs)), 4.5))
    for i, variant in enumerate(variants):
        r = at_k[at_k.variant == variant].set_index("dataset").reindex(logs)
        change = (r["mean"] / base.reindex(logs) - 1).values * 100
        err = (r["std"] / base.reindex(logs)).values * 100
        x = np.arange(len(logs)) + (i - (len(variants) - 1) / 2) * width
        shown = np.clip(change, -clip, clip)
        inside = np.abs(change) <= clip
        ax.errorbar(x[inside], shown[inside], yerr=err[inside], color=COLORS[variant], marker="o",
                    ms=5, capsize=2, lw=0, elinewidth=1, label=variant)
        for xi, v in zip(x[~inside & ~np.isnan(change)], change[~inside & ~np.isnan(change)]):
            ax.plot(xi, np.sign(v) * clip, marker="v" if v < 0 else "^", color=COLORS[variant], ms=7)
            ax.annotate(f"{v:+.0f}%", (xi, np.sign(v) * clip), fontsize=6, ha="center",
                        xytext=(0, -10 if v < 0 else 5), textcoords="offset points")
    ax.axhline(0, color=BASELINE_COLOR, ls="--", lw=1.2, label="F1+F2+B (baseline)")
    for xi in np.arange(len(logs) - 1) + 0.5:
        ax.axvline(xi, color="lightgrey", lw=0.5)
    ax.set_xticks(np.arange(len(logs)))
    ax.set_xticklabels(logs, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.5, len(logs) - 0.5)
    ax.set_ylim(-clip * 1.15, clip * 1.15)
    ax.set_ylabel("MAE change vs F1+F2+B (%)")
    ax.set_title(f"k = {k}, period selected on validation (negative = better than the baseline)")
    ax.legend(fontsize=7, loc="upper left", ncol=len(variants) + 1)
    fig.tight_layout()
    save_figure(fig, path)


def plot_k_selection(per_k, joint, variant, model_name, path, limit=10.0):
    """Per log: change of the test MAE vs the baseline with the smallest
    allowed k (light; 2 for F3, 1 for F4) and with k selected on the
    validation set (dark), with the selected k on the right. Bars beyond
    +-limit % are cut, with their value written inside."""
    light, dark, grid = "#9EC9E2", "#0072B2", "#E6E6E6"
    k_min = get_smallest_k(variant)
    fixed = per_k[(per_k.variant == variant) & (per_k.k == k_min)].set_index("dataset")["vs_B_%"]
    sel = joint[joint.variant == variant].set_index("dataset")
    logs = sorted(set(fixed.index) & set(sel.index), reverse=True)
    y, h = np.arange(len(logs)), 0.38
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(12.2 / 2.54, 9.5 / 2.54))
        for off, vals, color, label in [(h / 2, fixed.reindex(logs), light, f"$k = {k_min}$"),
                                        (-h / 2, sel["vs_B_%"].reindex(logs), dark,
                                         "$k$ selected on validation")]:
            ax.barh(y + off, vals.clip(-limit, limit), height=h, color=color, label=label,
                    edgecolor="white", linewidth=0.5)
            for yi, v in zip(y + off, vals):
                if abs(v) > limit:
                    ax.text(np.sign(v) * (limit - 0.3), yi, f"{v:.0f}%", va="center", fontsize=6,
                            ha="right" if v > 0 else "left",
                            color="white" if color == dark else "black")
        for yi, ds in zip(y, logs):
            ax.text(limit * 1.02, yi - h / 2, f"k={int(sel.loc[ds, 'k'])}", va="center",
                    fontsize=6.5, color="#555555")
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(logs)
        ax.set_xlim(-limit * 1.05, limit * 1.25)
        ax.set_xlabel("change of the test MAE vs. baseline F1+F2+B (%)   ← better | worse →")
        ax.grid(axis="x", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        n1 = int((fixed.reindex(logs) <= 0).sum())
        n2 = int((sel["vs_B_%"].reindex(logs) <= 0).sum())
        ax.set_title(f"{model_name}, {variant}   (MAE ≤ baseline: {n1}/{len(logs)} logs with $k={k_min}$, "
                     f"{n2}/{len(logs)} with $k$ selected)", loc="left", fontsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=7)
        save_figure(fig, path, pdf=True)


def plot_variant_pair(joint, variant_a, variant_b, label_a, label_b, model_name, path,
                      limit=10.0, characteristics=None):
    """Per log: change of the test MAE vs the baseline for two feature sets
    (light: variant_a, dark: variant_b), each with the period and k selected
    on the validation set; the selected k of both on the right. Bars beyond
    +-limit % are cut, with their value written inside. With characteristics
    (summarize_characteristics), a table on the right gives CV and variance
    fractions of each log."""
    light, dark, grid = "#9EC9E2", "#0072B2", "#E6E6E6"
    a = joint[joint.variant == variant_a].set_index("dataset")
    b = joint[joint.variant == variant_b].set_index("dataset")
    logs = sorted(set(a.index) & set(b.index), reverse=True)
    y, h = np.arange(len(logs)), 0.38
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(12.2 / 2.54, 11 / 2.54))
        if characteristics is not None:
            fig.subplots_adjust(right=0.68)
        for off, t, color, label in [(h / 2, a, light, label_a), (-h / 2, b, dark, label_b)]:
            vals = t["vs_B_%"].reindex(logs)
            ax.barh(y + off, vals.clip(-limit, limit), height=h, color=color, label=label,
                    edgecolor="white", linewidth=0.5)
            for yi, v in zip(y + off, vals):
                if abs(v) > limit:
                    ax.text(np.sign(v) * (limit - 0.3), yi, f"{v:.0f}%", va="center", fontsize=6,
                            ha="right" if v > 0 else "left",
                            color="white" if color == dark else "black")
            for yi, ds in zip(y + off, logs):
                ax.text(limit * 1.02, yi, f"k={int(t.loc[ds, 'k'])}", va="center",
                        fontsize=5.5, color="#555555")
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(logs)
        ax.set_xlim(-limit * 1.05, limit * 1.25)
        ax.set_xlabel("change of the test MAE vs. baseline F1+F2+B (%)   ← better | worse →")
        ax.grid(axis="x", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        na = int((a["vs_B_%"].reindex(logs) <= 0).sum())
        nb = int((b["vs_B_%"].reindex(logs) <= 0).sum())
        ax.set_title(f"{model_name}   (MAE ≤ baseline in {na}/{len(logs)} logs with {label_a}, "
                     f"{nb}/{len(logs)} {label_b})", loc="left", fontsize=7.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=7)
        if characteristics is not None:      # table of CV and variance fractions, one row per log
            heads = [("CV", "coefficient_of_variation"), ("$VF_T$", "vf_trend"),
                     ("$VF_S$", "vf_seasonal"), ("$VF_R$", "vf_residual")]
            for j, (head, col) in enumerate(heads):
                x = 1.17 + 0.105 * j
                ax.text(x, len(logs) - 0.3, head, transform=ax.get_yaxis_transform(),
                        ha="center", va="bottom", fontsize=6.5)
                for yi, ds in zip(y, logs):
                    v = characteristics[col].get(ds, np.nan)
                    ax.text(x, yi, "–" if pd.isna(v) else f"{v:.2f}",
                            transform=ax.get_yaxis_transform(), ha="center", va="center",
                            fontsize=6.5, color="#333333")
        save_figure(fig, path, pdf=True)


def plot_mae_per_log(table, model_name, path, ncols=3, from_zero=False):
    """One panel per log: test MAE in days (mean over the seeds, std as
    error bar) of the baseline and of each feature set with period and k
    selected on validation. Gray band: baseline mean +- one std. Each panel
    has its own x axis, since MAE levels differ by orders of magnitude; the
    change vs the baseline and the selected k are written next to each
    point. A legend below the panels explains the encoding. With from_zero,
    every x axis starts at 0 MAE, which shows the size of the differences
    relative to the error itself."""
    logs = sorted(table.dataset.unique())
    nrows = int(np.ceil(len(logs) / ncols))
    present = [v for v in VARIANTS if v[0] in set(table.variant)]
    style = dict(PAPER_STYLE, **{"font.size": 7, "axes.spines.left": False})
    with plt.rc_context(style):
        fig, axes = plt.subplots(nrows, ncols, figsize=(17 / 2.54, 2.6 / 2.54 * nrows), squeeze=False)
        for ax in axes.flat[len(logs):]:
            ax.axis("off")
        for ax, ds in zip(axes.flat, logs):
            t = table[table.dataset == ds].set_index("variant")
            b_mean, b_std = t.loc[BASELINE, ["mean", "std"]]
            ax.axvspan(b_mean - b_std, b_mean + b_std, color="#EEEEEE", zorder=0)
            ax.axvline(b_mean, color="#999999", lw=0.6, zorder=1)
            lo = 0.0 if from_zero else (t["mean"] - t["std"].fillna(0)).min()
            hi = (t["mean"] + t["std"].fillna(0)).max()
            pad = 0.12 * (hi - lo if hi > lo else abs(hi) or 1)
            for i, (variant, label, color, marker, _) in enumerate(present):
                if variant not in t.index:
                    continue
                r = t.loc[variant]
                y = len(present) - 1 - i
                ax.errorbar(r["mean"], y, xerr=r["std"], fmt=marker, color=color, ms=4,
                            elinewidth=0.9, capsize=2, zorder=3)
                if variant != BASELINE:
                    ax.text(hi + pad * 0.6, y, f"{r['vs_B_%']:+.1f}%  k={int(r['k'])}",
                            va="center", ha="left", fontsize=6, color="#333333")
            ax.set_xlim(0 if from_zero else lo - pad, hi + pad * (3.2 if from_zero else 6.5))
            ax.set_yticks(range(len(present)))
            ax.set_yticklabels([lab for _, lab, *_ in present][::-1])
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(-0.6, len(present) - 0.4)
            ax.set_title(ds, loc="left", fontsize=7.5, pad=2)
            ax.grid(axis="x", color="#E6E6E6", lw=0.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        scale = ", x axes from 0" if from_zero else ""
        fig.suptitle(f"{model_name}: test MAE in days per log (period and k selected on validation{scale})",
                     x=0.01, ha="left", fontsize=8)
        fig.tight_layout(h_pad=0.6, w_pad=1.2)
        handles = [plt.Line2D([], [], color=color, marker=marker, ms=4, ls="none", label=text)
                   for _, _, color, marker, text in present]
        handles += [plt.Line2D([], [], color="#555555", marker="|", ms=7, mew=0.9, lw=0.9,
                               label="error bar: ± one std over the seeds"),
                    Patch(facecolor="#EEEEEE", edgecolor="#999999", lw=0.6,
                          label="gray band: baseline mean ± one std"),
                    plt.Line2D([], [], ls="none", label="text: change of the mean MAE vs. B "
                                                        "(negative = better) and selected k")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=False, fontsize=6.5, handletextpad=0.5, columnspacing=2.0)
        save_figure(fig, path, pdf=True)


def plot_run(run, summary, per_k, joint, characteristics, out_dir):
    """All figures of one run."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(os.path.join(fig_dir, "by_k"), exist_ok=True)
    model_name = get_model_name(run)
    table = build_mae_table(joint, summary)
    plot_mae_per_log(table, model_name, os.path.join(fig_dir, "mae_per_log.png"))
    plot_mae_per_log(table, model_name, os.path.join(fig_dir, "mae_per_log_from_zero.png"),
                     from_zero=True)
    for variant in K_SELECTION_VARIANTS:
        if variant in set(per_k.variant):
            plot_k_selection(per_k, joint, variant, model_name,
                             os.path.join(fig_dir, f"k_selection_{variant.split('+')[-1]}.png"))
    for variant_a, variant_b, label_a, label_b in VARIANT_PAIRS:
        if {variant_a, variant_b} <= set(joint.variant):
            name = f"compare_{variant_a.split('+')[-1]}_vs_{variant_b.split('+')[-1]}.png"
            plot_variant_pair(joint, variant_a, variant_b, label_a, label_b, model_name,
                              os.path.join(fig_dir, name), characteristics=characteristics)
    plot_selected(joint, summary, os.path.join(fig_dir, "selected.png"))
    plot_per_k(per_k, summary, os.path.join(fig_dir, "per_k.png"))
    plot_per_k(per_k, summary, os.path.join(fig_dir, "per_k_with_selection.png"), joint=joint,
               title="Curve: period selected for each k.  Star: (period, k) selected jointly.")
    plot_wins_per_k(count_wins(per_k, summary, by=["k"]), os.path.join(fig_dir, "wins_per_k.png"))
    plot_all_periods(summary, os.path.join(fig_dir, "all_periods.png"))
    for k in sorted(per_k.k.unique()):
        plot_vs_baseline_for_k(per_k, summary, k, os.path.join(fig_dir, "by_k", f"k{k:02d}.png"))


# ===================================================== 3. characteristics of the logs

def list_prepared_logs():
    """Logs with a prepared time series file, without the excluded ones."""
    suffix = "_series.pkl"
    names = sorted(f[:-len(suffix)] for f in os.listdir(PREP_DIR) if f.endswith(suffix))
    return [n for n in names if n not in EXCLUDED_LOGS]


def describe_log(ds, period=7, q_level=0.2):
    """One row per time series of a log: statistics and variance fractions
    on the training period (after warm-up removal, before the first test
    case starts), so no test information is used."""
    with open(os.path.join(PREP_DIR, f"{ds}_series.pkl"), "rb") as f:
        s = pickle.load(f)
    rows = []
    for ser in select_train_period(s["series"], s["test_start"], q_level):
        row = {"dataset": ds, "series": ser.name,
               "train_start": ser.index.min(), "train_end": ser.index.max()}
        row.update(analyze_series_stats(ser))
        row.update(compute_variance_fractions(ser, period))
        rows.append(row)
    return rows


def load_characteristics(refresh):
    """Per log: CV and variance fractions of trend, seasonal and residual,
    averaged over its three time series. Computed and saved on first use
    (or with refresh)."""
    if refresh or not os.path.exists(CHARACTERISTICS_FILE):
        rows = [row for ds in list_prepared_logs() for row in describe_log(ds)]
        pd.DataFrame(rows).to_csv(CHARACTERISTICS_FILE, index=False)
        print("written:", CHARACTERISTICS_FILE)
    table = pd.read_csv(CHARACTERISTICS_FILE)
    cols = ["coefficient_of_variation", "vf_trend", "vf_seasonal", "vf_residual"]
    return table.groupby("dataset")[cols].mean()


# ===================================================== 4. table of the test MAE

def build_mae_csv(mae_tables, names):
    """The table of the paper: one row per log; per run and feature set the
    mean and std of the test MAE in days over the seeds, and per run the
    feature set with the lowest mean ("best", decided on the unrounded
    values). The number of logs in which a feature set is at least as good
    as B is in <run>/wins.csv (at_least_as_good)."""
    common = set.intersection(*(set(t.dataset) for t in mae_tables))
    logs = [ds for ds in LOG_ORDER if ds in common] + sorted(common - set(LOG_ORDER))
    table = pd.DataFrame(index=pd.Index(logs, name="log"))
    for t, name in zip(mae_tables, names):
        t = t.set_index(["dataset", "variant"])
        means = pd.DataFrame(index=logs)
        for v, label, *_ in VARIANTS:
            if v not in t.index.get_level_values("variant"):
                continue
            mean = t["mean"].xs(v, level="variant").reindex(logs)
            table[f"{name} {label} mean"] = mean
            table[f"{name} {label} std"] = t["std"].xs(v, level="variant").reindex(logs)
            means[label] = mean
        table[f"{name} best"] = means.idxmin(axis=1)
    numeric = table.select_dtypes("number").columns    # 6 significant digits (sepsis: ~0.01 days)
    table[numeric] = table[numeric].map(lambda x: float(f"{x:.6g}") if pd.notna(x) else x)
    return table


# ===================================================== 5. time series of the logs

def plot_time_series(ds, path):
    """The daily time series of one log; the dashed line is the start of
    the test period."""
    with open(os.path.join(PREP_DIR, f"{ds}_series.pkl"), "rb") as f:
        s = pickle.load(f)
    fig, axes = plt.subplots(len(s["series"]), 1, figsize=(12, 2.2 * len(s["series"])),
                             sharex=True, squeeze=False)
    for ax, ser in zip(axes[:, 0], s["series"]):
        ax.plot(ser.index, ser.values, color="#0072B2", lw=0.8)
        ax.axvline(s["test_start"], color="grey", ls="--", lw=1)
        ax.set_ylabel(ser.name, fontsize=9)
    fig.suptitle(ds)
    fig.tight_layout()
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["xgb_periods", "rf_periods"],
                        help="runs in results/runs/ (one column group of the table each)")
    parser.add_argument("--exclude", nargs="+", default=[], help="logs left out (besides EXCLUDED_LOGS)")
    parser.add_argument("--refresh-characteristics", action="store_true",
                        help="compute the characteristics of the logs again")
    parser.add_argument("--no-time-series", action="store_true", help="skip the time series figures")
    args = parser.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.set_option("display.width", 220)

    characteristics = load_characteristics(args.refresh_characteristics)
    mae_tables, names = [], []
    for run in args.runs:
        out_dir = os.path.join(REPORT_DIR, run)
        os.makedirs(out_dir, exist_ok=True)
        print(f"{run}: selection on validation ...")
        summary, per_k, joint = run_selection(run, args.exclude, out_dir)
        print(f"{run}: figures ...")
        plot_run(run, summary, per_k, joint, characteristics, out_dir)
        mae_tables.append(build_mae_table(joint, summary))
        names.append(get_model_name(run))
        wins = count_wins(joint, summary, by=[]).set_index("variant")
        print(wins[["at_least_as_good", "clear_wins", "n_logs", "median_vs_B_pct", "sign_test_p"]]
              .round(3).to_string())
        print("written:", out_dir, "\n")

    table = build_mae_csv(mae_tables, names)
    table.to_csv(TABLE_FILE)
    print("written:", TABLE_FILE)

    if not args.no_time_series:
        ts_dir = os.path.join(REPORT_DIR, "time_series")
        os.makedirs(ts_dir, exist_ok=True)
        for ds in list_prepared_logs():
            plot_time_series(ds, os.path.join(ts_dir, f"{ds}.png"))
        print("written:", ts_dir)


if __name__ == "__main__":
    main()