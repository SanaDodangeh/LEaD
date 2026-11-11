"""Hyperparameter selection on the validation set and retraining on the whole
training set, for XGBoost and the random forest.

Protocol (Verenich et al.):
  1. tuning   every hyperparameter combination of the model is trained, for
              every configuration, seasonal period p, number of lags k and
              seed, on the training cases (without the validation cases)
  2. --select per log and configuration, the setting (p, k, hyperparameters)
              with the lowest validation MAE, averaged over the seeds
  3. --refit  the selected setting is retrained on the whole training set
              (training and validation cases) and evaluated on the test set

Grids (Verenich et al., Table 9; random forest: max_features only):
    xgb   n_estimators, learning_rate, subsample, colsample_bytree, max_depth  (48)
    rf    max_features in {0.33, 1.0}; the other settings of experiment.yaml   (2)
XGBoost and the random forest are trained with a fixed number of trees. 

Configurations (variant names of experiment_lag.py; Bagg from run_bagg_baseline.py):
    F1+F2+B      baseline (snapshot)            no p, no k
Configurations (variant names of experiment_lag.py; Bagg3 and Bindex from run_bagg_baseline.py):
    F1+F2+Bagg3  Base-agg: B at each event, aggregated over the prefix      no p, no k
    F1+F2+Bindex Base-index: B at each event, one column per event         no p, no k
    F1+F2+F3     ND: lags of the time series    k
    F1+F2+F4     LEaD                           p and k
    F1+F2+F4ts   NR: LEaD without residual      p and k


Results go to evaluation/results/runs/<run name>/ (default: tune_<model>):
all_logs.csv (tuning; a stopped run continues where it stopped), selected.csv
and mae_table.csv (--select), refit.csv and mae_table_refit.csv (--refit).
Nothing of the existing code or runs is changed.

Usage:
    python evaluation/hp_tune.py --model xgb
    python evaluation/hp_tune.py --model xgb --select
    python evaluation/hp_tune.py --model xgb --refit
    python evaluation/hp_tune.py --model rf
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import experiment_lag as E
from experiment_lag import R, D
from run_bagg_baseline import load_bagg
from scipy.stats import rankdata

GRIDS = {
    "xgb": {                       # Verenich et al., Table 9
        "n_estimators": [250, 500],
        "learning_rate": [0.02, 0.04, 0.06],
        "subsample": [0.5, 0.8],
        "colsample_bytree": [0.5, 0.8],
        "max_depth": [3, 6],
    },
    "rf": {
        "max_features": [0.33, 1.0],
    },
    "lstm": {                      # Verenich et al., Table 9 (activation: ReLU, fixed)
        "units": [100, 200],
        "layers": [1, 2, 3],
        "batch_size": [8, 32],
        "optimizer": ["rmsprop", "nadam"],
    },
}
INT_PARAMS = {"n_estimators", "max_depth", "units", "layers", "batch_size"}
VARIANTS = ["F1+F2+B", "F1+F2+Bagg3", "F1+F2+Bindex", "F1+F2+F3", "F1+F2+F4"]
#FRIEDMAN_VARIANTS = ["F1+F2+B", "F1+F2+Bagg3", "F1+F2+F3", "F1+F2+F4"]  
WILCOXON_PAIRS = [("F1+F2+F3", "F1+F2+B"), ("F1+F2+F3", "F1+F2+Bagg3"), ("F1+F2+F3", "F1+F2+Bindex"),
                  ("F1+F2+F4", "F1+F2+B"), ("F1+F2+F4", "F1+F2+Bagg3"), ("F1+F2+F4", "F1+F2+Bindex"),
                  ("F1+F2+F4", "F1+F2+F3")]
DEFAULT_SEEDS = {"xgb": [10, 11, 12, 13, 14], "rf": [0, 1, 2, 3, 4], "lstm": [0, 1, 2, 3, 4]}


def list_hyperparameters(model):
    """All combinations of the grid of a model, as dicts."""
    grid = GRIDS[model]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def cast_hyperparameters(hp):
    """Hyperparameters read from a CSV file back to their types."""
    return {c: (int(v) if c in INT_PARAMS else v) for c, v in hp.items()}


def get_keys(model):
    return ["dataset", "variant", "period", "k"] + list(GRIDS[model]) + ["seed"]

######################################################################################
# training
######################################################################################

def train_xgb(X_fit, y_fit, X_val, X_test, hp, seed, n_jobs):
    """XGBoost on log1p(remaining seconds) with a fixed number of trees;
    returns the predictions for validation and test (seconds)."""
    model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed,
                             n_jobs=n_jobs, **hp)
    model.fit(X_fit, np.log1p(y_fit), verbose=False)
    pred_val = np.clip(np.expm1(model.predict(X_val)), 0, None) if len(X_val) else np.zeros(0)
    pred_test = np.clip(np.expm1(model.predict(X_test)), 0, None)
    return pred_val, pred_test


def train_model(model, data, X, train, val, test, hp, seed, args, exp_cfg, epochs=None):
    """Train one model on the rows train and predict val and test (seconds).
    Returns (pred_val, pred_test, best_epoch); best_epoch only for the LSTM."""
    y = data["y"]
    if model == "xgb":
        pv, pt = train_xgb(X[train], y[train], X[val], X[test], hp, seed, args.n_jobs)
        return pv, pt, np.nan
    if model == "rf":
        params = dict(exp_cfg["rf"], name="rf", n_jobs=args.n_jobs, **hp)
        X_val = X[val] if val.any() else X[test][:1]
        pt, _, info = train_and_predict(X[train], y[train], X_val, y[val], X[test], params, seed)
        return (info["pred_val"] if val.any() else np.zeros(0)), pt, np.nan
    params = dict(exp_cfg["lstm"], gpu=False, **hp)
    if epochs is not None:
        params["fixed_epochs"] = int(epochs)
    extra = prepare_extra_features(X, train)
    pt, _, info = train_and_predict_lstm(data["seq"], extra, y, train, val, test, params, seed)
    return info["pred_val"], pt, info["best_epoch"]


#####################################################################################
# data
#####################################################################################

def load_log(ds, model, variants, exp_cfg, ds_cfg):
    """Feature table, time series, splits, target, Bagg and (LSTM) event sequences of a log."""
    tbl = pd.read_pickle(os.path.join(E.PREP_DIR, f"{ds}_table.pkl"))
    data = {"tbl": tbl, "s": E.load_series(ds), "tau": tbl[D.TS].values.astype("datetime64[ns]"),
            "y": tbl[D.TARGET].to_numpy()}
    for x in ("train", "val", "test"):
        data[x] = (tbl[D.SET] == x).values
    data["bagg3"] = load_bagg(ds, tbl, exp_cfg, ds_cfg, "Bagg3") if any("Bagg3" in v.split("+") for v in variants) else None
    data["bindex"] = load_bagg(ds, tbl, exp_cfg, ds_cfg, "Bindex") if any("Bindex" in v.split("+") for v in variants) else None
    data["seq"] = E.load_sequences(ds, tbl, exp_cfg) if model == "lstm" else None
    return data


def build_matrix(model, variant, k, data, components):
    """Feature matrix of a configuration; Bagg3 and Bindex are added to the
    groups of experiment_lag; the LSTM reads the events instead of F1."""
    groups = [g for g in variant.split("+") if not (model == "lstm" and g == "F1")]
    X = E.build_inputs("+".join(g for g in groups if g not in ("Bagg3", "Bindex")), k, data["tbl"], data["s"],
                       data["tau"], components)
    if "Bagg3" in groups:
        X = pd.concat([X, data["bagg3"].astype(np.float32)], axis=1)
    if "Bindex" in groups:
        X = pd.concat([X, data["bindex"].astype(np.float32)], axis=1)
    return X.to_numpy(np.float32)


def list_settings(ds, variant, periods, ks, reuse):
    """(period, k) pairs of a configuration: from the run reuse, or all of them
    (p only with F4/F4ts, k only with a windowed group)."""
    if not E.is_windowed(variant):
        return [(0, 0)]
    if reuse is not None:
        r = reuse.loc[(ds, variant)]
        return [(int(r["period"]), int(r["k"]))]
    if E.uses_period(variant):
        return [(p, k) for p in E.get_allowed_periods(ds, periods) for k in ks]
    return [(0, k) for k in ks]


def load_reuse(run):
    """(period, k) selected per log and configuration in another run."""
    if not run:
        return None
    return pd.read_csv(os.path.join(E.RUNS_DIR, run, "selected.csv")).set_index(["dataset", "variant"])


def load_done(path, keys):
    if not os.path.exists(path):
        return set()
    prev = pd.read_csv(path)
    return set(map(tuple, prev[keys].astype(str).values))


def append_row(path, row):
    """Append one result to a CSV file; the columns follow the header of an
    existing file, so that files written by an earlier version stay aligned."""
    if os.path.exists(path):
        columns = pd.read_csv(path, nrows=0).columns
        frame = pd.DataFrame([row]).reindex(columns=columns)
        frame.to_csv(path, mode="a", index=False, header=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)

############################################################################################
# 1 tuning
############################################################################################


def tune_log(ds, args, exp_cfg, ds_cfg, path, reuse):
    """Train every configuration, setting, hyperparameter combination and seed of one log."""
    keys, hp_cols = get_keys(args.model), list(GRIDS[args.model])
    done = load_done(path, keys)
    data = load_log(ds, args.model, args.variants, exp_cfg, ds_cfg)
    y_val, y_test = data["y"][data["val"]] / R.SECONDS_PER_DAY, data["y"][data["test"]] / R.SECONDS_PER_DAY
    hps = list_hyperparameters(args.model)
    for variant in args.variants:
        for period, k in list_settings(ds, variant, args.periods, args.ks, reuse):
            todo = [(hp, seed) for hp in hps for seed in args.seeds
                    if tuple(map(str, [ds, variant, period, k] + [hp[c] for c in hp_cols] + [seed])) not in done]
            if not todo:
                continue
            components = E.load_components(ds, period, max(max(args.ks), k)) if period else None
            X = build_matrix(args.model, variant, k, data, components)
            start = time.time()
            for hp, seed in todo:
                pv, pt, best_epoch = train_model(args.model, data, X, data["train"], data["val"], data["test"],
                                                 hp, seed, args, exp_cfg)
                row = {"dataset": ds, "variant": variant, "period": period, "k": k, **hp, "seed": seed,
                       "n_features": X.shape[1], "best_epoch": best_epoch,
                       "val_mae": float(np.mean(np.abs(pv / R.SECONDS_PER_DAY - y_val))),
                       "mae": float(np.mean(np.abs(pt / R.SECONDS_PER_DAY - y_test)))}
                append_row(path, row)
            E.log(f"{ds:14s} {variant:12s} p={period:2d} k={k:2d}  {len(todo)} fits in {time.time() - start:.0f}s")


############################################################################################
# 2 selection
############################################################################################


def select_settings(run_dir, args):
    """Per log and configuration: the setting (p, k, hyperparameters) with the
    lowest validation MAE averaged over the seeds. Only the values of k in
    args.ks are considered (k = 0: configurations without lags). With
    args.hp_variant, the hyperparameters are selected on that configuration
    and used for all configurations, with (p, k) from args.reuse_pk."""
    hp_cols = list(GRIDS[args.model])
    res = pd.read_csv(os.path.join(run_dir, "all_logs.csv")).drop_duplicates(get_keys(args.model))
    res = res[res["dataset"] != "dataset"]                       # header lines of parallel processes
    for col in ("period", "k", "seed"):
        res[col] = res[col].astype(int)
    res = res[res["seed"].isin(args.seeds)]                      # only the seeds given with --seeds
    res = res[res["variant"] != "F1+F2+Bagg"]       # old aggregated baseline (active cases only), replaced by Bagg3
    for col in ("val_mae", "mae"):
        res[col] = res[col].astype(float)
    if "best_epoch" not in res.columns:                          # runs made before this column existed
        res["best_epoch"] = np.nan
    res["best_epoch"] = res["best_epoch"].astype(float)
    if args.reuse_pk is None:
        res = res[res["k"].isin(list(args.ks) + [0])]
    if args.hp_variant and args.reuse_pk is None and any(E.is_windowed(v) for v in args.variants):
        raise SystemExit("--hp-variant needs --reuse-pk for configurations with lags")
    group = ["dataset", "variant", "period", "k"] + hp_cols
    summary = res.groupby(group).agg(val_mae=("val_mae", "mean"), mean=("mae", "mean"), std=("mae", "std"),
                                     n_seeds=("seed", "nunique"),
                                     epochs=("best_epoch", "mean")).reset_index()
    if not args.hp_variant:
        selected = summary.loc[summary.groupby(["dataset", "variant"])["val_mae"].idxmin()]
    else:
        reuse = load_reuse(args.reuse_pk)
        rows = []
        base = summary[summary["variant"] == args.hp_variant]
        for ds, b in base.groupby("dataset"):
            best = b.loc[b["val_mae"].idxmin()]
            for variant in args.variants:
                period, k = list_settings(ds, variant, args.periods, args.ks, reuse)[0]
                match = summary[(summary["dataset"] == ds) & (summary["variant"] == variant)
                                & (summary["period"] == period) & (summary["k"] == k)]
                for c in hp_cols:
                    match = match[match[c] == best[c]]
                row = match.iloc[0].to_dict() if len(match) else {
                    "dataset": ds, "variant": variant, "period": period, "k": k,
                    **{c: best[c] for c in hp_cols}, "val_mae": np.nan, "mean": np.nan, "std": np.nan,
                    "n_seeds": 0, "epochs": np.nan}
                rows.append(row)
        selected = pd.DataFrame(rows)
    selected.to_csv(os.path.join(run_dir, "selected.csv"), index=False)
    table = selected.pivot(index="dataset", columns="variant", values=["mean", "std"])
    table.to_csv(os.path.join(run_dir, "mae_table.csv"))
    with pd.option_context("display.width", 250, "display.precision", 3):
        print(selected[["dataset", "variant", "period", "k"] + hp_cols + ["val_mae", "mean", "std", "n_seeds"]]
              .to_string(index=False))
    print("written:", os.path.join(run_dir, "selected.csv"), "and mae_table.csv")


############################################################################################
# 3 retraining
############################################################################################


def refit_selected(run_dir, args, exp_cfg, ds_cfg):
    """Retrain the selected setting of every log and configuration on the
    whole training set (training and validation cases) and evaluate it on
    the test set. The LSTM is trained for the number of epochs at which the
    selected setting was best on validation (mean over the tuning seeds); if
    that setting was not tuned (--hp-variant), this number is determined by
    one run with early stopping first. Writes refit.csv and mae_table_refit.csv."""
    hp_cols = list(GRIDS[args.model])
    selected = pd.read_csv(os.path.join(run_dir, "selected.csv"))
    path = os.path.join(run_dir, "refit.csv")
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = set(zip(prev.dataset, prev.variant, prev.seed))
    for ds, rows in selected.groupby("dataset", sort=False):
        rows = rows[rows["variant"].isin(args.variants)]          # only the configurations of --variants
        if rows.empty or (args.datasets and ds not in args.datasets):
            continue
        data = load_log(ds, args.model, list(rows.variant), exp_cfg, ds_cfg)
        whole = data["train"] | data["val"]                     # the whole training set
        no_val = np.zeros_like(whole)
        y_test = data["y"][data["test"]] / R.SECONDS_PER_DAY
        for _, r in rows.iterrows():
            todo = [seed for seed in args.seeds if (ds, r.variant, seed) not in done]
            if not todo:
                continue
            period, k = int(r.period), int(r.k)
            components = E.load_components(ds, period, max(max(args.ks), k)) if period else None
            X = build_matrix(args.model, r.variant, k, data, components)
            hp = cast_hyperparameters({c: r[c] for c in hp_cols})
            epochs = None
            if args.model == "lstm":
                epochs = r.get("epochs", np.nan)
                if pd.isna(epochs):                              # not tuned: early stopping run on validation
                    _, _, epochs = train_model("lstm", data, X, data["train"], data["val"], data["test"],
                                               hp, args.seeds[0], args, exp_cfg)
                epochs = max(1, int(round(epochs)))
            for seed in todo:
                _, pt, _ = train_model(args.model, data, X, whole, no_val, data["test"], hp, seed, args,
                                       exp_cfg, epochs)
                row = {"dataset": ds, "variant": r.variant, "period": period, "k": k, **hp, "seed": seed,
                       "epochs": epochs if epochs is not None else np.nan,
                       "mae": float(np.mean(np.abs(pt / R.SECONDS_PER_DAY - y_test)))}
                append_row(path, row)
            E.log(f"{ds:14s} {r.variant:12s} retrained on the whole training set ({len(todo)} seeds)")
    res = pd.read_csv(path)
    res = res[res["dataset"] != "dataset"].drop_duplicates(["dataset", "variant", "seed"])
    res["mae"] = res["mae"].astype(float)
    table = res.groupby(["dataset", "variant"])["mae"].agg(["mean", "std"]).unstack("variant")
    table.to_csv(os.path.join(run_dir, "mae_table_refit.csv"))
    test_configurations(res, run_dir)
    test_lead_pairs(res, run_dir)
    with pd.option_context("display.width", 250, "display.precision", 3):
        print(table.to_string())
    print("written:", path, "and mae_table_refit.csv")

def test_configurations(res, run_dir, variants=FRIEDMAN_VARIANTS):
    """Friedman test over the logs: do the configurations differ in test MAE
    (mean over the seeds)? Average rank per configuration (1 = lowest MAE of
    the log; ties get the average rank, Demsar 2006). Logs without all
    configurations are left out. Writes friedman.csv to the run folder."""
    from scipy.stats import friedmanchisquare
    mae = res.groupby(["dataset", "variant"])["mae"].mean().unstack()
    variants = [v for v in variants if v in mae.columns]
    t = mae[variants].dropna()
    stat, p = friedmanchisquare(*[t[v] for v in variants])
    out = pd.DataFrame({"variant": variants, "average_rank": t.rank(axis=1).mean()[variants].values,
                        "n_logs": len(t), "chi2": stat, "p": p})
    out.to_csv(os.path.join(run_dir, "friedman.csv"), index=False)
    print(f"Friedman test over {len(t)} logs: chi2 = {stat:.2f}, p = {p:.3f}")
    print(out[["variant", "average_rank"]].round(2).to_string(index=False))
    
def test_pairs_wilcoxon(res, run_dir, pairs=WILCOXON_PAIRS, alpha=0.05, name="wilcoxon", alternative="greater"):
    """Wilcoxon signed-rank test over the logs on the gain (in %) of the
    configuration over the reference per log, computed from the mean MAE;
    one-sided by default (alternative: the gain is positive). effect_size:
    rank-biserial correlation (> 0: the configuration is better)."""
    from scipy.stats import wilcoxon
    mae = res.astype({"mae": float}).groupby(["dataset", "variant"])["mae"].mean().unstack()
    rows = []
    for a, b in pairs:
        if a not in mae.columns or b not in mae.columns:
            continue
        t = mae[[a, b]].dropna()
        gain = (100 * (t[b] - t[a]) / t[b]).to_numpy()        # gain of the variant over the reference per log, in %
        stat, p = wilcoxon(gain, alternative=alternative)      # greater: the gain is positive (scale-free)
        from scipy.stats import rankdata
        d = gain
        d = d[d != 0]                                   # zero differences are dropped, as in the test
        r = rankdata(np.abs(d))
        effect = (r[d > 0].sum() - r[d < 0].sum()) / r.sum()    # rank-biserial; > 0: variant better
        rows.append({"variant": a, "reference": b, "n_logs": len(t), "n_better": int((t[a] < t[b]).sum()),
                     "median_gain": float((100 * (t[b] - t[a]) / t[b]).median()), "statistic": stat, "p": p, "effect_size": effect})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    out["significant"] = out["p"] < alpha
    out.to_csv(os.path.join(run_dir, f"{name}.csv"), index=False)
    print(f"{name}: Wilcoxon signed-rank, alternative = {alternative}, without correction (alpha = {alpha})")    
    print(out.round(3).to_string(index=False))
    return out



def test_lead_pairs(res, run_dir):
    """Two sets of Wilcoxon tests without correction: (1) all five pairs,
    (2) LEaD-full against LEaD-raw."""
    test_pairs_wilcoxon(res, run_dir, name="wilcoxon_all")
    test_pairs_wilcoxon(res, run_dir, [("F1+F2+F4", "F1+F2+F3")], name="wilcoxon_full_vs_raw")

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "rf", "lstm"], required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="default: all prepared logs, largest first")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--periods", nargs="+", type=int, default=[7, 30])
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 7, 14])
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 10-14 (xgb), 0-4 (rf, lstm)")
    parser.add_argument("--n-jobs", type=int, default=8, help="CPU threads per XGBoost or random forest model")
    parser.add_argument("--run-name", default=None, help="default: tune_<model>")
    parser.add_argument("--reuse-pk", default=None, metavar="RUN",
                        help="take (p, k) from the selection of this run (e.g. tune_xgb)")
    parser.add_argument("--hp-variant", default=None, metavar="VARIANT",
                        help="with --select: select the hyperparameters on this configuration only")
    parser.add_argument("--select", action="store_true", help="only select the settings from the results so far")
    parser.add_argument("--refit", action="store_true",
                        help="retrain the selected settings on the whole training set and evaluate them on the test set")
    parser.add_argument("--friedman", action="store_true",
                        help="only the Friedman test on refit.csv (no training)")
    args = parser.parse_args()
    args.seeds = args.seeds or DEFAULT_SEEDS[args.model]

    run_dir = os.path.join(E.RUNS_DIR, args.run_name or f"tune_{args.model}")
    os.makedirs(run_dir, exist_ok=True)
    if args.friedman:
        res = pd.read_csv(os.path.join(run_dir, "refit.csv"))
        res = res[res["dataset"] != "dataset"].drop_duplicates(["dataset", "variant", "seed"]).astype({"mae": float})
        test_configurations(res, run_dir)
        test_lead_pairs(res, run_dir)
        return
    if args.select:
        select_settings(run_dir, args)
        return
    exp_cfg, ds_cfg = R.load_config("experiment.yaml"), R.load_config("datasets.yaml")
    if args.refit:
        refit_selected(run_dir, args, exp_cfg, ds_cfg)
        return
    datasets = args.datasets or [ds for ds in E.list_datasets()
                                 if os.path.exists(os.path.join(E.PREP_DIR, f"{ds}_table.pkl"))]
    reuse = load_reuse(args.reuse_pk)
    E.log(f"run folder: {run_dir}; model {args.model}; {len(list_hyperparameters(args.model))} hyperparameter "
          f"combinations; p {args.periods}; k {args.ks}; reuse (p, k): {args.reuse_pk}; seeds {args.seeds}; "
          f"{len(datasets)} logs: {datasets}")
    for ds in datasets:
        tune_log(ds, args, exp_cfg, ds_cfg, os.path.join(run_dir, "all_logs.csv"), reuse)
    E.log("done")


if __name__ == "__main__":
    main()