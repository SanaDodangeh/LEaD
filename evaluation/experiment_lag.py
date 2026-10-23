"""Train and evaluate all feature sets on all logs.

  1. prepare each log (feature table with F1, F2, B; daily time series for
     F3; their expanding-window STL decomposition for F4), saved in
     data/prepared/
  2. train F1+F2+B once, and F1+F2+F3, F1+F2+F4 and F1+F2+F4ts for every
     window k (F4 and F4ts also for every seasonal period); the test and
     validation MAE of every setting and seed are saved

Each run writes to its own folder evaluation/results/runs/<run name>/ (default name:
date and time of the start):
    all_logs.csv      one row per log, feature set, period, k and seed
    comparison.csv    mean and std over the seeds per log, feature set, period and k
    run_info.yaml     settings of the first start of the run
Every result is appended to all_logs.csv at once; results that are already
there are skipped, so a run started again with the same --run-name
continues where it stopped. Period and k are selected on the validation
set, and all figures and tables are made, by report.py

Usage:
    python evaluation/experiment_lag.py --periods 7 14 30 --seeds 10 11 12 13 14 --run-name xgb_periods
    python evaluation/experiment_lag.py --model rf --periods 7 14 30 --run-name rf_periods
    python evaluation/experiment_lag.py --datasets helpdesk --variants F1+F2+B F1+F2+F4ts --ks 1 2 3 --seeds 0 1
    python evaluation/experiment_lag.py --compare xgb_periods rf_periods --names XGB RF
    python evaluation/experiment_lag.py --compare xgb_periods rf_periods --names XGB RF --compare-k F1+F2+F3 F1+F2+F4
"""
import argparse
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import yaml

import run_experiment as R
from dyRtp import data as D
from dyRtp import verenich as V
from dyRtp.features import F3_F4_process as P
from dyRtp.features.baseline_snapshot import build_snapshot_features
from dyRtp.model import (build_sequence_data, prepare_extra_features, train_and_predict,
                         train_and_predict_lstm)
import report as S

warnings.simplefilter("ignore", RuntimeWarning)   # NaN windows at the start of a log

RAW_DIR = os.path.join(R.ROOT, "data", "raw")
PREP_DIR = os.path.join(R.ROOT, "data", "prepared")
RUNS_DIR = os.path.join(R.EVAL_DIR, "results", "runs")
# set by set_run_dir: every run writes to its own folder results/runs/<run name>/
RUN_DIR = RESULTS_FILE = COMPARISON_FILE = None
ALL_VARIANTS = ["F1+F2+B", "F1+F2+F3", "F1+F2+F4", "F1+F2+F4ts"]
# feature groups encoded with a window k: F3 (observed lags), F4 (decomposed
# lags) and F4ts (decomposed lags without the residual)
WINDOWED_GROUPS = ("F3", "F4", "F4ts")
MIN_CYCLES = 6     # a seasonal period is used only if the fit period covers this many cycles
EXCLUDE = ["credit", "invoice"]    # credit: no resource attribute; invoice: left out
INTERVAL = "1D"


def set_run_dir(run_name):
    """Use results/runs/<run_name>/ for the results and comparison tables of
    this run. Runs that share a run name (parallel processes, or a restart
    after an interruption) share the folder and skip finished results."""
    global RUN_DIR, RESULTS_FILE, COMPARISON_FILE
    RUN_DIR = os.path.join(RUNS_DIR, run_name)
    RESULTS_FILE = os.path.join(RUN_DIR, "all_logs.csv")
    COMPARISON_FILE = os.path.join(RUN_DIR, "comparison.csv")
    os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)


def uses_period(variant):
    """True if the variant contains F4 or F4ts, whose STL decomposition
    depends on the seasonal period."""
    groups = variant.split("+")
    return "F4" in groups or "F4ts" in groups


def is_windowed(variant):
    """True if the variant contains a group encoded with a window k."""
    return any(g in WINDOWED_GROUPS for g in variant.split("+"))


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def list_datasets():
    """All logs of the benchmark whose CSV file is in data/raw, largest file
    first (the long runs start first, the short ones fill the gaps at the end)."""
    paths = {ds: os.path.join(RAW_DIR, os.path.basename(f)) for ds, f in V.vconf.filename.items()
             if ds not in EXCLUDE}
    found = [ds for ds, path in paths.items() if os.path.exists(path)]
    return sorted(found, key=lambda ds: os.path.getsize(paths[ds]), reverse=True)


def prepare_log(ds, exp_cfg, ds_cfg, max_k):
    """Build and save the feature table (F1, F2, B) and the time series
    (F3) with their decomposition (F4) of one log."""
    tbl = R.build_feature_table(ds, exp_cfg, logger=log).reset_index(drop=True)
    ov = exp_cfg["verenich_overrides"]
    dm = V.get_manager(ds, RAW_DIR, ov["drop_columns"], ov["rename_columns"])
    data = dm.read_dataset()
    cc, tc, rc = dm.case_id_col, dm.timestamp_col, ds_cfg[ds]["resource_col"]
    tau = tbl[D.TS].values.astype("datetime64[ns]")

    B = build_snapshot_features(data, cc, rc, tc, tau, INTERVAL)
    tbl = pd.concat([tbl, B], axis=1)

    grid = P.build_interval_grid(data[tc], INTERVAL)
    series = [P.compute_active_cases(data, cc, tc, grid),
              P.compute_arrival_rate(data, cc, tc, grid),
              P.compute_active_resources(data, rc, tc, grid)]
    components = P.decompose_all_time_series(series, max_k=max_k)

    test_start = tbl.loc[tbl[D.SET] == "test", D.TS].min()
    tbl.to_pickle(os.path.join(PREP_DIR, f"{ds}_table.pkl"))
    with open(os.path.join(PREP_DIR, f"{ds}_series.pkl"), "wb") as f:
        pickle.dump({"series": series, "components": components, "test_start": test_start}, f)


def load_series(ds):
    """Time series, components and test start of one prepared log."""
    with open(os.path.join(PREP_DIR, f"{ds}_series.pkl"), "rb") as f:
        return pickle.load(f)


def get_fit_days(ds, tbl=None):
    """Number of days from the start of the log's time series to the start
    of the validation period (= days available to fit the models). Cached
    in data/prepared/<log>_fit_days.txt."""
    path = os.path.join(PREP_DIR, f"{ds}_fit_days.txt")
    if os.path.exists(path):
        with open(path) as f:
            return int(f.read())
    if tbl is None:
        tbl = pd.read_pickle(os.path.join(PREP_DIR, f"{ds}_table.pkl"))
    first_day = load_series(ds)["series"][0].index.min()
    val_start = tbl.loc[tbl[D.SET] == "val", D.TS].min()
    days = int((val_start - first_day) / pd.Timedelta("1D"))
    with open(path, "w") as f:
        f.write(str(days))
    return days


def get_allowed_periods(ds, periods, tbl=None):
    """Seasonal periods of which the fit period covers at least MIN_CYCLES
    cycles. Longer periods leave F4 empty for a large part of the fit
    period (STL needs two cycles of history) and are not estimated well."""
    days = get_fit_days(ds, tbl)
    return [p for p in periods if days >= MIN_CYCLES * p]


def load_components(ds, period, max_k):
    """F4 components of a log for one seasonal period, with at least max_k
    lags. Period 7 with up to 14 lags is part of the prepared time series;
    everything else is computed on first use and saved as
    data/prepared/<log>_components_p<period>.pkl."""
    s = load_series(ds)
    if period == 7 and min(W.shape[1] for W in s["components"].values()) >= max_k:
        return s["components"]
    path = os.path.join(PREP_DIR, f"{ds}_components_p{period}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            components = pickle.load(f)
        if min(W.shape[1] for W in components.values()) >= max_k:
            return components
    log(f"{ds}: decomposing with period {period} ...")
    components = P.decompose_all_time_series(s["series"], max_k=max_k, period=period)
    with open(path, "wb") as f:
        pickle.dump(components, f)
    return components


def build_inputs(variant, k, tbl, s, tau, components=None):
    """Feature matrix of a variant; F3, F4 and F4ts are lag-encoded with
    window k. components: F4 components to use (default: those of period 7)."""
    groups = variant.split("+")
    static = "+".join(g for g in groups if g in ("F1", "F2", "B"))
    parts = [tbl[R.select_columns(static, tbl)]]
    if "F3" in groups:
        parts.append(pd.concat([P.encode_lag(ser, tau, k) for ser in s["series"]], axis=1))
    if "F4" in groups or "F4ts" in groups:
        components = s["components"] if components is None else components
        if "F4ts" in groups:        # F4 without the residual: trend and seasonal only
            components = {n: W for n, W in components.items() if not n.endswith("_residual")}
        parts.append(pd.concat([P.encode_decomposed_lag(n, W, tau, k)
                                for n, W in components.items()], axis=1))
    return pd.concat(parts, axis=1).astype(np.float32)


def load_sequences(ds, tbl, exp_cfg):
    """Event sequences of the prefixes of one log, for the LSTM (see
    dyRtp/model.py). Cached in data/prepared/<log>_sequences.pkl."""
    path = os.path.join(PREP_DIR, f"{ds}_sequences.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    ov = exp_cfg["verenich_overrides"]
    dm = V.get_manager(ds, RAW_DIR, ov["drop_columns"], ov["rename_columns"])
    seq = build_sequence_data(dm, dm.read_dataset(), tbl, D.CASE, D.EVENT_NR, D.SET,
                                exp_cfg["lstm"]["max_levels"])
    with open(path, "wb") as f:
        pickle.dump(seq, f)
    log(f"{ds}: sequences ready ({seq['events'].shape[1]} features per event)")
    return seq


def load_done():
    """(dataset, variant, period, k, seed) of the results already saved."""
    if not os.path.exists(RESULTS_FILE):
        return set()
    prev = pd.read_csv(RESULTS_FILE)
    if "period" not in prev.columns:
        prev["period"] = 0
    return set(zip(prev.dataset, prev.variant, prev.period, prev.k, prev.seed))


def append_result(row):
    pd.DataFrame([row]).to_csv(RESULTS_FILE, mode="a", index=False,
                               header=not os.path.exists(RESULTS_FILE))


def run_variant(ds, variant, ks, seeds, model_cfg, periods=(7,)):
    """Train and evaluate one variant on one log, for each seasonal period
    (only variants with F4), window k and seed. Both the test MAE and the
    validation MAE are saved; the validation MAE is used to select the
    period and k. Results already in RESULTS_FILE are skipped."""
    if uses_period(variant):
        allowed = get_allowed_periods(ds, periods)
        if len(allowed) < len(periods):
            log(f"{ds}: periods {sorted(set(periods) - set(allowed))} skipped "
                f"(fit period of {get_fit_days(ds)} days < {MIN_CYCLES} cycles)")
        periods = allowed
    else:
        periods = [0]
    done = load_done()
    if all((ds, variant, p, k, seed) in done for p in periods for k in ks for seed in seeds):
        return
    tbl = pd.read_pickle(os.path.join(PREP_DIR, f"{ds}_table.pkl"))
    s = load_series(ds)
    tau = tbl[D.TS].values.astype("datetime64[ns]")
    fit = (tbl[D.SET] == "train").values
    val = (tbl[D.SET] == "val").values
    test = (tbl[D.SET] == "test").values
    y = tbl[D.TARGET].to_numpy()
    y_val_days = y[val] / R.SECONDS_PER_DAY
    y_test_days = y[test] / R.SECONDS_PER_DAY
    use_lstm = model_cfg["name"] == "lstm"
    if use_lstm:          # the event sequence replaces the case features F1
        seq = load_sequences(ds, tbl, R.load_config("experiment.yaml"))
        variant_inputs = "+".join(g for g in variant.split("+") if g != "F1")
    else:
        variant_inputs = variant

    for period in periods:
        components = load_components(ds, period, max(ks)) if period else None
        for k in ks:
            todo = [seed for seed in seeds if (ds, variant, period, k, seed) not in done]
            if not todo:
                continue
            X = build_inputs(variant_inputs, k, tbl, s, tau, components)
            if use_lstm:
                X = prepare_extra_features(X, fit)
            for seed in todo:
                if use_lstm:
                    pred, _, info = train_and_predict_lstm(seq, X, y, fit, val, test, model_cfg, seed)
                else:
                    pred, _, info = train_and_predict(X[fit], y[fit], X[val], y[val], X[test],
                                                      model_cfg, seed)
                mae = float(np.mean(np.abs(pred / R.SECONDS_PER_DAY - y_test_days)))
                val_mae = float(np.mean(np.abs(info["pred_val"] / R.SECONDS_PER_DAY - y_val_days)))
                append_result({"dataset": ds, "variant": variant, "period": period, "k": k,
                               "seed": seed, "n_features": X.shape[1], "mae": mae,
                               "val_mae": val_mae, "trees": info["best_iteration"]})
                log(f"{ds:14s} {variant:12s} p={period:2d} k={k:2d} seed={seed}  "
                    f"MAE {mae:.3f}  (val {val_mae:.3f})")


def summarize_results():
    """Write comparison.csv from the results saved so far: per log,
    feature set, period and k, the mean and std of the test MAE (days) over
    the seeds, the mean validation MAE, and the change of the test MAE in %
    relative to F1+F2 and to the baseline F1+F2+B (negative = better).
    Period and k are selected on the validation set by report.py."""
    if not os.path.exists(RESULTS_FILE):
        return
    results = pd.read_csv(RESULTS_FILE)
    for col, default in [("period", 0), ("val_mae", np.nan)]:
        if col not in results.columns:
            results[col] = default
    results["period"] = results["period"].fillna(0).astype(int)
    results = results.drop_duplicates(["dataset", "variant", "period", "k", "seed"])
    # drop periods that are too long for the log (also from runs before this rule)
    too_long = [ds for ds in results.dataset.unique()
                if os.path.exists(os.path.join(PREP_DIR, f"{ds}_fit_days.txt"))]
    for ds in too_long:
        days = get_fit_days(ds)
        results = results[~((results.dataset == ds) & (results.period * MIN_CYCLES > days))]
    summary = (results.groupby(["dataset", "variant", "period", "k"], sort=False)
               .agg(mean=("mae", "mean"), std=("mae", "std"), n_seeds=("mae", "count"),
                    val_mean=("val_mae", "mean"))
               .reset_index())
    ref = summary[summary.k == 0].groupby(["dataset", "variant"])["mean"].first().unstack()
    for name, col in [("F1+F2", "vs_F1+F2_%"), ("F1+F2+B", "vs_B_%")]:
        base = summary.dataset.map(ref[name]) if name in ref else np.nan
        summary[col] = (summary["mean"] / base - 1) * 100
    summary.to_csv(COMPARISON_FILE, index=False)


def load_selected_per_k(run_name, exclude):
    """Period selected on the validation set per log, feature set and k (as
    in report.py), with the change vs the baseline; and the
    summary over the seeds (for the std of the baseline)."""
    results = S.load_results(run_name)
    results = results[~results.dataset.isin(EXCLUDE + exclude)
                      & ~results.variant.isin(S.EXCLUDED_VARIANTS)]
    summary = S.summarize_over_seeds(results)
    per_k = S.add_change_vs_references(S.select_on_validation(summary, by=["k"]), summary)
    return per_k, summary


def restrict_to_common(tables):
    """Keep only the logs and feature sets that every run contains."""
    logs = set.intersection(*(set(per_k.dataset) for per_k, _ in tables))
    variants = set.intersection(*(set(per_k.variant) for per_k, _ in tables))
    return [(per_k[per_k.dataset.isin(logs) & per_k.variant.isin(variants)],
             summary[summary.dataset.isin(logs)]) for per_k, summary in tables], sorted(logs)


def summarize_wins_per_k(per_k, summary, name):
    """Per feature set and k: logs at least as good as the baseline, clear
    wins (better by more than one std of the baseline), median change in %,
    sign test p-value."""
    wins = S.count_wins(per_k, summary, by=["k"]).set_index(["variant", "k"])
    return pd.DataFrame({
        f"{name} >=B": wins["at_least_as_good"].astype(str) + "/" + wins["n_logs"].astype(str),
        f"{name} clear": wins["clear_wins"],
        f"{name} med%": wins["median_vs_B_pct"].round(1),
        f"{name} p": wins["sign_test_p"].round(3)})


def summarize_wins_per_log(per_k, name):
    """Per log and feature set: number of k with MAE at most the baseline,
    and the median change over the k in %."""
    g = per_k.groupby(["dataset", "variant"])
    return pd.DataFrame({
        f"{name} k>=B": (g["vs_B_%"].apply(lambda x: (x <= 0).sum()).astype(str)
                         + "/" + g["k"].count().astype(str)),
        f"{name} med%": g["vs_B_%"].median().round(1)})


def compare_runs(runs, names, exclude):
    """Print the results of several runs (e.g. XGBoost and random forest)
    side by side, on the logs and feature sets that all runs contain."""
    names = names or runs
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    tables, logs = restrict_to_common([load_selected_per_k(r, exclude) for r in runs])
    print(f"{len(logs)} logs in all runs: {logs}\n")
    t1 = pd.concat([summarize_wins_per_k(p, s, n) for (p, s), n in zip(tables, names)], axis=1)
    print("Table 1: per feature set and k (period selected on validation)")
    for variant, block in t1.groupby(level="variant"):
        print(f"\n{variant}\n" + block.droplevel("variant").to_string())
    t2 = pd.concat([summarize_wins_per_log(p, n) for (p, _), n in zip(tables, names)], axis=1)
    print("\nTable 2: per log, number of k with MAE <= baseline, and median change over k (%)")
    for variant, block in t2.groupby(level="variant"):
        print(f"\n{variant}\n" + block.droplevel("variant").to_string())


def compare_k_selection(runs, names, exclude, variants):
    """The smallest allowed k (MIN_K_RAW = 2 for F3, 1 for F4 and F4ts) vs k
    selected on the validation set, per log, feature set and run. For the
    smallest k the period is selected on validation (as for every k); for
    the selected k, the pair (period, k) with the lowest mean validation MAE
    is used. Changes are vs the baseline F1+F2+B in % (negative = better).
    Only logs that all runs contain are compared."""
    names = names or runs
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    loaded = {}
    for run, name in zip(runs, names):
        per_k, summary = load_selected_per_k(run, exclude)
        joint = S.add_change_vs_references(S.select_on_validation(summary, by=[]), summary)
        loaded[name] = (per_k, joint)
    logs = sorted(set.intersection(*(set(p.dataset) for p, _ in loaded.values())))
    print(f"{len(logs)} logs in all runs: {logs}")
    for variant in variants:
        raw_only = "F3" in variant and "F4" not in variant     # as in report.py
        k_min = S.MIN_K_RAW if raw_only else 1
        for name, (per_k, joint) in loaded.items():
            fixed = per_k[(per_k.variant == variant) & (per_k.k == k_min)].set_index("dataset")
            sel = joint[joint.variant == variant].set_index("dataset")
            if fixed.empty:
                print(f"\n{name} {variant}: not in this run")
                continue
            t = pd.DataFrame({f"k={k_min} vs B %": fixed["vs_B_%"], "selected k": sel["k"],
                              "selected period": sel["period"], "selected vs B %": sel["vs_B_%"],
                              "fixed MAE": fixed["mean"], "selected MAE": sel["mean"]}).reindex(logs)
            t["better"] = np.where(t["fixed MAE"] < t["selected MAE"], f"k={k_min}",
                                   np.where(t["fixed MAE"] > t["selected MAE"], "selected", "same"))
            print(f"\n{name} {variant}")
            print(t.drop(columns=["fixed MAE", "selected MAE"]).round(2).to_string())
            print(f"  logs with MAE <= B:  k={k_min} {int((t[f'k={k_min} vs B %'] <= 0).sum())}/{len(t)},  "
                  f"selected k {int((t['selected vs B %'] <= 0).sum())}/{len(t)}")
            print(f"  median change vs B:  k={k_min} {t[f'k={k_min} vs B %'].median():.1f}%,  "
                  f"selected k {t['selected vs B %'].median():.1f}%")
            print(f"  lower test MAE:      k={k_min} in {int((t.better == f'k={k_min}').sum())} logs,  "
                  f"selected k in {int((t.better == 'selected').sum())} logs,  "
                  f"same in {int((t.better == 'same').sum())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None, help="default: all logs in data/raw")
    parser.add_argument("--ks", nargs="+", type=int, default=list(range(1, 15)))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--periods", nargs="+", type=int, default=[7],
                        help="seasonal periods (days) of the STL decomposition, for F4")
    parser.add_argument("--prepare-only", action="store_true",
                        help="only prepare the logs and the F4 components of all periods, then stop")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        help="feature sets, run in this order (default: B, F3, F4 and F4ts)")
    parser.add_argument("--run-name", default=None,
                        help="folder name in results/runs/ (default: current date and time); "
                             "use the same name to continue a run or for parallel processes")
    parser.add_argument("--model", choices=["xgb", "rf", "lstm"], default="xgb",
                        help="xgb: XGBoost (config 'model'); rf: random forest (config 'rf'); "
                             "lstm: LSTM on the event sequence (config 'lstm')")
    parser.add_argument("--reuse-selection", default=None, metavar="RUN",
                        help="train only the (period, k) selected on validation in this run "
                             "(e.g. xgb_periods), per log and feature set, instead of all of them")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="only compare these finished runs side by side, then stop (trains nothing)")
    parser.add_argument("--names", nargs="+", default=None, help="with --compare: short names of the runs")
    parser.add_argument("--compare-exclude", nargs="+", default=[],
                        help="with --compare: logs left out (e.g. an unfinished log)")
    parser.add_argument("--compare-k", nargs="+", default=None, metavar="VARIANT",
                        help="with --compare: instead, compare the smallest allowed k (2 for F3, "
                             "1 for F4 and F4ts) with k selected on validation for these feature sets")
    args = parser.parse_args()
    if args.compare and args.compare_k:
        compare_k_selection(args.compare, args.names, args.compare_exclude, args.compare_k)
        return
    if args.compare:
        compare_runs(args.compare, args.names, args.compare_exclude)
        return
    set_run_dir(args.run_name or time.strftime("%Y-%m-%d_%H-%M"))

    exp_cfg = R.load_config("experiment.yaml")
    ds_cfg = R.load_config("datasets.yaml")
    datasets = args.datasets or list_datasets()
    seeds = args.seeds if args.seeds is not None else exp_cfg["seeds"]
    model_cfg = dict(exp_cfg[{"xgb": "model", "rf": "rf", "lstm": "lstm"}[args.model]], name=args.model)
    os.makedirs(PREP_DIR, exist_ok=True)
    log(f"run folder: {RUN_DIR}")
    log(f"{len(datasets)} logs: {datasets}")
    log(f"model: {args.model}; feature sets: {args.variants}; k: {args.ks}; "
        f"periods: {args.periods}; seeds: {seeds}")
    info_file = os.path.join(RUN_DIR, "run_info.yaml")
    if not os.path.exists(info_file):    # settings of the run, for reproducibility
        with open(info_file, "w") as f:
            yaml.safe_dump({"variants": args.variants, "ks": args.ks, "seeds": seeds,
                            "periods": args.periods, "reuse_selection": args.reuse_selection,
                            "model": model_cfg}, f)

    ready = []
    for ds in datasets:
        if not os.path.exists(os.path.join(PREP_DIR, f"{ds}_series.pkl")):
            try:
                prepare_log(ds, exp_cfg, ds_cfg, max(args.ks))
            except Exception as e:
                log(f"{ds}: skipped ({type(e).__name__}: {e})")
                continue
        ready.append(ds)
        log(f"{ds}: ready")

    if args.prepare_only:
        for ds in ready:
            for period in get_allowed_periods(ds, args.periods):
                load_components(ds, period, max(args.ks))
            log(f"{ds}: components ready for periods {get_allowed_periods(ds, args.periods)} "
                f"(fit period {get_fit_days(ds)} days)")
        log("done")
        return

    selected = None
    if args.reuse_selection:      # (period, k) per log and feature set, from report.py
        path = os.path.join(S.REPORT_DIR, args.reuse_selection, "selected.csv")
        selected = pd.read_csv(path).set_index(["dataset", "variant"])
        log(f"(period, k) selected on validation in {args.reuse_selection}: {path}")
    plan = [(v, args.ks if is_windowed(v) else [0]) for v in args.variants]
    for variant, ks in plan:
        for ds in ready:
            if selected is not None and is_windowed(variant):
                if (ds, variant) not in selected.index:
                    log(f"{ds} {variant}: not in {args.reuse_selection}, skipped")
                    continue
                r = selected.loc[(ds, variant)]
                run_variant(ds, variant, [int(r["k"])], seeds, model_cfg,
                            [int(r["period"])] if r["period"] else args.periods)
            else:
                run_variant(ds, variant, ks, seeds, model_cfg, args.periods)
        summarize_results()
        log(f"{variant} finished; comparison tables updated")
    log("done")


if __name__ == "__main__":
    main()