"""Feature table of a log, used by experiment_lag.py: split, prefixes and
labels from the benchmark of Verenich et al., and the feature groups F1
(case) and F2 (calendar); B, F3 and F4 are added by experiment_lag.py.

Run on its own, it trains XGBoost on F1 and F1+F2 of one log (a check of
the feature table):
    python evaluation/run_experiment.py --dataset helpdesk

Outputs of the stand-alone run (in evaluation/results/<dataset>/):
    metrics.csv          one row per variant and seed
    predictions.csv.gz   true and predicted remaining time per prefix
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))   # evaluation/: scripts and results/
ROOT = os.path.dirname(EVAL_DIR)                         # repository root: dyRtp/, configs/, data/
sys.path.insert(0, ROOT)                                 # so that dyRtp is found

from dyRtp import data as D
from dyRtp import verenich as V
from dyRtp.utilities import compute_trimming_window
from dyRtp.features.F1_case import CaseFeatures, add_event_columns
from dyRtp.features.F2_calendar import compute_calendar_features, check_log_spans_year

# the benchmark code builds frames column by column; harmless on newer pandas
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

SECONDS_PER_DAY = 86400.0


def load_config(name):
    """Read a YAML file from configs/."""
    with open(os.path.join(ROOT, "configs", name)) as f:
        return yaml.safe_load(f)


def build_feature_table(dataset, exp_cfg, logger=print):
    """Return one row per prediction point: meta columns plus all feature groups.

    Split, prefixes and labels use the benchmark code (dyRtp/verenich.py);
    each feature group comes from its module in dyRtp/features/."""
    t0 = time.time()
    ov = exp_cfg["verenich_overrides"]
    dm = V.get_manager(dataset, os.path.join(ROOT, "data", "raw"),
                       ov["drop_columns"], ov["rename_columns"])
    data = dm.read_dataset()
    data = V.add_case_attributes(dm, data)
    data = add_event_columns(dm, data)
    ts = dm.timestamp_col
    logger(f"[{dataset}] {data[dm.case_id_col].nunique()} cases, {len(data)} events, "
           f"{data[ts].min():%Y-%m-%d} to {data[ts].max():%Y-%m-%d}")

    sp, pf = exp_cfg["split"], exp_cfg["prefixes"]
    train, test = dm.split_data(data, sp["train_ratio"])
    fit, val = dm.split_data(train, 1 - sp["val_ratio"])
    max_len = min(pf["max_length_cap"],
                  dm.get_pos_case_length_quantile(data, pf["max_length_quantile"]))
    include_month = check_log_spans_year(data.rename(columns={ts: D.TS}))

    # warm-up and cool-down removal: keep prediction points within the window
    tr = exp_cfg.get("trimming")
    if tr:
        win_start, win_end = compute_trimming_window(data, dm.case_id_col, ts, tr["interval"], tr["q_level"])
        logger(f"[{dataset}] trimming: keep tau in ({win_start:%Y-%m-%d}, {win_end:%Y-%m-%d}]")

    case_features = CaseFeatures(dm)
    parts = []
    for name, part in [("train", fit), ("val", val), ("test", test)]:  # train first: fixes F1 columns
        prefixes = dm.generate_prefix_data(part, pf["min_length"], max_len)
        info = V.get_prefix_meta(dm, prefixes)
        meta = pd.DataFrame({
            D.PREFIX_ID: info.index,
            D.CASE: info["case"].values,
            D.EVENT_NR: info["event_nr"].values,
            D.TS: info["tau"].values,
            D.TARGET: V.get_labels(dm, prefixes).reindex(info.index).values,
            D.SET: name,
        }, index=info.index)

        # prediction points to keep
        if pf["drop_complete_traces"]:
            keep = info["event_nr"] < info["orig_case_length"]
        else:
            keep = pd.Series(True, index=info.index)
        if tr:
            keep &= (info["tau"] > win_start) & (info["tau"] <= win_end)

        F1 = case_features.encode_prefixes(prefixes)
        F2 = compute_calendar_features(meta[[D.TS]], include_month)
        parts.append(pd.concat([meta, F1, F2], axis=1).loc[keep])
        logger(f"[{dataset}] {name}: {int(keep.sum())} prefixes")

    table = pd.concat(parts, ignore_index=True)
    n1 = sum(c.startswith("F1__") for c in table.columns)
    n2 = sum(c.startswith("F2__") for c in table.columns)
    logger(f"[{dataset}] max prefix length {max_len}; F1 {n1} cols, F2 {n2} cols "
           f"(month: {include_month}); {time.time()-t0:.1f}s")
    return table


def select_columns(variant, table):
    """Feature columns of the groups in a variant such as "F1+F2"."""
    groups = variant.split("+")
    return [c for c in table.columns if c.split("__")[0] in groups]


def predict_median_baseline(table):
    """Median remaining time of training prefixes with the same length."""
    train = table[table[D.SET] == "train"]
    med = train.groupby(D.EVENT_NR)[D.TARGET].median()
    test = table[table[D.SET] == "test"]
    return test[D.EVENT_NR].map(med).fillna(train[D.TARGET].median()).to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variants", nargs="+", default=["F1", "F1+F2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="recompute the feature table")
    args = parser.parse_args()

    from dyRtp.model import train_and_predict  # imported here so --help works without xgboost

    exp_cfg = load_config("experiment.yaml")
    seeds = args.seeds if args.seeds is not None else exp_cfg["seeds"]

    cache_dir = os.path.join(ROOT, "data", "features")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{args.dataset}.pkl")
    if os.path.exists(cache) and not args.rebuild:
        table = pd.read_pickle(cache)
        print(f"[{args.dataset}] loaded cached feature table {table.shape}")
    else:
        table = build_feature_table(args.dataset, exp_cfg)
        table.to_pickle(cache)

    fit = table[D.SET] == "train"
    val = table[D.SET] == "val"
    test = table[D.SET] == "test"
    y = table[D.TARGET].to_numpy()
    y_test_days = y[test] / SECONDS_PER_DAY

    out_dir = os.path.join(EVAL_DIR, "results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    preds = table.loc[test, [D.CASE, D.EVENT_NR, D.TS]].copy()
    preds["y_true_days"] = y_test_days
    rows = []

    base = predict_median_baseline(table) / SECONDS_PER_DAY
    preds["median_baseline"] = base
    rows.append({"dataset": args.dataset, "variant": "median_baseline", "seed": -1,
                 "n_features": 0, "mae_days": float(np.mean(np.abs(base - y_test_days)))})
    print(f"median baseline: MAE {rows[-1]['mae_days']:.3f} days")

    for variant in args.variants:
        cols = select_columns(variant, table)
        X = table[cols].astype(np.float32)
        for seed in seeds:
            pred, _, info = train_and_predict(
                X[fit], y[fit], X[val], y[val], X[test], exp_cfg["model"], seed)
            pred_days = pred / SECONDS_PER_DAY
            mae = float(np.mean(np.abs(pred_days - y_test_days)))
            preds[f"{variant}_seed{seed}"] = pred_days
            rows.append({"dataset": args.dataset, "variant": variant, "seed": seed,
                         "n_features": len(cols), "mae_days": mae, **info})
            print(f"{variant:8s} seed {seed}: MAE {mae:.3f} days "
                  f"({len(cols)} features, {info['best_iteration']} trees, "
                  f"{info['train_time_s']:.1f}s)")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    preds.to_csv(os.path.join(out_dir, "predictions.csv.gz"), index=False)

    summary = metrics.groupby("variant", sort=False)["mae_days"].agg(["mean", "std", "count"])
    print("\nMAE in days (mean, std over seeds):")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()