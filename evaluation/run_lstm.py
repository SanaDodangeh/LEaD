"""LSTM of the benchmark of Verenich et al., with the benchmark's own event
features (including open_cases at every event), on the prefixes of this study.

Purpose: compare the tree-based models with our features against the LSTM
as the benchmark runs it. Only the data protocol is ours, so that the MAEs
are comparable with the other runs:
  - prefixes, split and labels: data/prepared/<log>_table.pkl (the same
    train, validation and test prefixes as all other runs: warm-up and
    cool-down removed, complete traces dropped)
  - tuning with one seed on the training cases, early stopping on the
    validation cases; the setting with the lowest validation MAE is
    retrained on training + validation cases for its best number of epochs
    with five seeds and evaluated on the test prefixes (as hp_tune.py)

From the benchmark (experiments/train_LSTM.py, DatasetManager_LSTM.py,
run_LSTM.sh of Verenich et al.), unchanged in content:
  - event features: all columns of dataset_confs.py, including open_cases
    and the calendar columns; numeric columns min-max scaled, activity and
    the other categorical columns one-hot encoded (all values seen in
    training), prefixes padded at the front to the maximum prefix length
  - architecture: 1-3 LSTM layers with BatchNormalization, Dense(1) output
    with the activation of the grid, dropout 0.15, learning rate 0.001,
    Nadam (with clipvalue 3) or RMSprop, mean absolute error, early stopping
    after 42 epochs, learning rate halved after 10 epochs (at most 500)
  - grid: Verenich et al., Table 9, as hp_tune.py: units {100, 200},
    layers {1, 2, 3}, batch size {8, 32}, optimizer {RMSprop, Nadam};
    activation ReLU

Changes needed to run it (no change of the method):
  - tf.keras instead of Keras 2.0 (argument names: learning_rate instead of
    lr, min_delta instead of epsilon; Nadam's schedule_decay 0.004 is the
    built-in default); the best epoch is restored with
    restore_best_weights instead of checkpoint files
  - the label is the remaining time in days (--unit seconds: as the
    benchmark's CSV); MAE is reported in days in both cases
  - in the retraining there is no validation set, so the learning rate is
    reduced on the training loss
  - missing numeric values (if any) are set to 0 after scaling

Results: evaluation/results/runs/verenich_lstm/ tuning.csv, selected.csv,
refit.csv (variant "Verenich-LSTM"). Finished rows are skipped, so a stopped
run continues where it stopped.

Usage (environment with tensorflow, pandas, scikit-learn, pyyaml):
    python evaluation/run_lstm.py --datasets helpdesk
    python evaluation/run_lstm.py --datasets helpdesk --part 0 --n-parts 3   (grid split over 3 processes)
    python evaluation/run_lstm.py --datasets helpdesk --refit
    python evaluation/run_lstm.py --datasets helpdesk --calendar           (with F2 as event input)
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, ROOT)
from dyRtp import verenich as V                                  # noqa: E402
from dyRtp.data import CASE, EVENT_NR, TARGET, SET               # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "raw")
PREP_DIR = os.path.join(ROOT, "data", "prepared")
VARIANT = "Verenich-LSTM"
GRID = {"units": [100, 200], "layers": [1, 2, 3], "batch_size": [8, 32], "optimizer": ["rmsprop", "nadam"]}
FIXED = {"activation": "relu", "dropout": 0.15, "learning_rate": 0.001,
         "max_epochs": 500, "patience": 42, "lr_patience": 10}
SECONDS_PER_DAY = 86400.0
BENCHMARK_CALENDAR = ["month", "weekday", "hour", "timesincemidnight"]   # replaced by F2 with --calendar


# ---------------------------------------------------------------- data

def load_events(ds, calendar, intercase=False):
    """Events of a log with all columns of dataset_confs.py, open_cases
    included (only the renaming of experiment.yaml is applied). With
    calendar, the calendar columns of the benchmark are replaced by F2 of
    each event (hour, day of week and, for logs longer than a year, month;
    as for the tree-based models), so that F2 of tau is the last step. 
    With intercase, new cases and active resources in the day
    before each event (the definitions of B) are added."""
    with open(os.path.join(ROOT, "configs", "experiment.yaml")) as f:
        rename = yaml.safe_load(f)["verenich_overrides"]["rename_columns"]
    dm = V.get_manager(ds, RAW_DIR, drop_columns=BENCHMARK_CALENDAR if calendar else [], rename_columns=rename)
    data = dm.read_dataset()
    data = data.sort_values(dm.timestamp_col, kind="mergesort")
    data = data.sort_values([dm.case_id_col, dm.timestamp_col], kind="mergesort").reset_index(drop=True)
    extra = []
    if calendar:
        ts = data[dm.timestamp_col]
        data["F2__hour"], data["F2__day_of_week"] = ts.dt.hour.astype(float), ts.dt.dayofweek.astype(float)
        extra = ["F2__hour", "F2__day_of_week"]
        if ts.max() - ts.min() > pd.Timedelta(days=365):
            data["F2__month"] = ts.dt.month.astype(float)
            extra.append("F2__month")
    if intercase:                          # new cases and active resources at each event, as B (open_cases: benchmark)
        from dyRtp.features.baseline_snapshot import count_recent_arrivals, count_recent_resources
        with open(os.path.join(ROOT, "configs", "datasets.yaml")) as f:
            res_col = yaml.safe_load(f)[ds]["resource_col"]
        cc, tc = dm.case_id_col, dm.timestamp_col
        t = data[tc].values.astype("datetime64[ns]")
        data["B__arrivals"] = count_recent_arrivals(data, cc, tc, t, "1D").astype(float)
        data["B__active_resources"] = count_recent_resources(data, res_col, tc, t, "1D").astype(float)
        extra += ["B__arrivals", "B__active_resources"]
    return dm, data, extra


def encode_events(dm, data, train_cases, extra_num):
    """DatasetManager_LSTM.encode_data of the benchmark: numeric columns
    (and extra_num) min-max scaled, activity and the other categorical
    columns one-hot encoded; scaler and one-hot columns from the events of
    train_cases."""
    from sklearn.preprocessing import MinMaxScaler
    num_cols = list(dict.fromkeys(dm.dynamic_num_cols + dm.static_num_cols + extra_num))
    cat_cols = [c for c in dict.fromkeys(dm.dynamic_cat_cols + dm.static_cat_cols) if c != dm.activity_col]
    train_rows = data[dm.case_id_col].astype(str).isin(train_cases).values
    parts = []
    if num_cols:
        num = data[num_cols].astype(float).values
        scaler = MinMaxScaler().fit(num[train_rows])
        parts.append(pd.DataFrame(scaler.transform(num), columns=num_cols))
    parts.append(pd.get_dummies(data[dm.activity_col], prefix="act").astype(float))
    if cat_cols:
        parts.append(pd.get_dummies(data[cat_cols]).astype(float))
    enc = pd.concat(parts, axis=1)
    seen = [c for c in enc.columns if c in num_cols or enc.loc[train_rows, c].sum() > 0]
    enc = enc[seen]
    n_missing = int(enc.isna().sum().sum())
    if n_missing:
        print(f"  {n_missing} missing values set to 0 after scaling")
    return enc.fillna(0).to_numpy(np.float32), seen


def build_sequences(dm, data, events, prefixes, max_len):
    """One matrix per prefix (rows: its events, padded with zeros at the
    front to max_len, as pad_sequences in the benchmark)."""
    case_ids = data[dm.case_id_col].astype(str).values
    starts = pd.Series(np.arange(len(data))).groupby(case_ids).min()
    first = starts.reindex(prefixes[CASE].astype(str).values).values
    if np.isnan(first.astype(float)).any():
        raise ValueError("prefixes of cases that are not in the log")
    X = np.zeros((len(prefixes), max_len, events.shape[1]), dtype=np.float32)
    for i, (s, n) in enumerate(zip(first.astype(int), prefixes[EVENT_NR].astype(int).values)):
        X[i, max_len - n:] = events[s:s + n]
    return X


def load_log(ds, unit, calendar, intercase=False):
    """Feature table (prefixes, sets, labels) and raw events of a log."""
    tbl = pd.read_pickle(os.path.join(PREP_DIR, f"{ds}_table.pkl"))
    dm, data, extra = load_events(ds, calendar, intercase)
    y = tbl[TARGET].to_numpy(float) / (SECONDS_PER_DAY if unit == "days" else 1.0)
    sets = {s: (tbl[SET] == s).values for s in ("train", "val", "test")}
    return {"tbl": tbl, "dm": dm, "data": data, "extra": extra, "y": y, "sets": sets,
            "max_len": int(tbl[EVENT_NR].max()), "scale": 1.0 if unit == "days" else SECONDS_PER_DAY}


def build_inputs(log, train_mask):
    """Sequences of all prefixes, encoded with the events of the cases in train_mask."""
    tbl = log["tbl"]
    train_cases = set(tbl.loc[train_mask, CASE].astype(str))
    events, cols = encode_events(log["dm"], log["data"], train_cases, log["extra"])
    print(f"  {len(cols)} features per event, prefixes up to {log['max_len']} events")
    return build_sequences(log["dm"], log["data"], events, tbl, log["max_len"])


# ---------------------------------------------------------------- model

def build_model(n_steps, n_features, hp):
    """train_LSTM.py of the benchmark, in tf.keras."""
    from tensorflow import keras
    L = keras.layers
    act, dropout, units = FIXED["activation"], FIXED["dropout"], hp["units"]
    main_input = L.Input(shape=(n_steps, n_features), name="main_input")
    if hp["layers"] == 1:
        l2_3 = L.LSTM(units, kernel_initializer="glorot_uniform", return_sequences=False, dropout=dropout)(main_input)
        b2_3 = L.BatchNormalization()(l2_3)
    elif hp["layers"] == 2:
        l1 = L.LSTM(units, activation=act, kernel_initializer="glorot_uniform", return_sequences=True,
                    dropout=dropout)(main_input)
        b1 = L.BatchNormalization(axis=1)(l1)
        l2_3 = L.LSTM(units, activation=act, kernel_initializer="glorot_uniform", return_sequences=False,
                      dropout=dropout)(b1)
        b2_3 = L.BatchNormalization()(l2_3)
    else:
        l1 = L.LSTM(units, kernel_initializer="glorot_uniform", return_sequences=True, dropout=dropout)(main_input)
        b1 = L.BatchNormalization(axis=1)(l1)
        l2 = L.LSTM(units, kernel_initializer="glorot_uniform", return_sequences=True, dropout=dropout)(b1)
        b2 = L.BatchNormalization(axis=1)(l2)
        l3 = L.LSTM(units, kernel_initializer="glorot_uniform", return_sequences=False, dropout=dropout)(b2)
        b2_3 = L.BatchNormalization()(l3)
    out = L.Dense(1, activation=act, kernel_initializer="glorot_uniform", name="outcome_output")(b2_3)
    model = keras.Model(inputs=[main_input], outputs=[out])
    if hp["optimizer"] == "nadam":         # "adam" in train_LSTM.py: Nadam with clipvalue 3
        opt = keras.optimizers.Nadam(learning_rate=FIXED["learning_rate"], beta_1=0.9, beta_2=0.999,
                                     epsilon=1e-08, clipvalue=3)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=FIXED["learning_rate"], rho=0.9, epsilon=1e-08)
    model.compile(loss="mean_absolute_error", optimizer=opt)
    return model


def set_seed(seed, threads):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:                    # threads can be set only before the first operation
        pass


def train_with_validation(X, y, sets, hp, seed, threads):
    """Training cases, early stopping on the validation cases; returns the
    validation MAE and the epoch of the best weights, and the model."""
    from tensorflow import keras
    set_seed(seed, threads)
    model = build_model(X.shape[1], X.shape[2], hp)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=FIXED["patience"], restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=FIXED["lr_patience"],
                                            min_delta=0.0001, cooldown=0, min_lr=0)]
    tr, va = sets["train"], sets["val"]
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), batch_size=hp["batch_size"],
                     epochs=FIXED["max_epochs"], callbacks=cb, verbose=0)
    val = np.array(hist.history["val_loss"])
    return float(val.min()), int(val.argmin() + 1), len(val), model


def train_fixed_epochs(X, y, mask, hp, epochs, seed, threads):
    """Training + validation cases for a fixed number of epochs (no validation set)."""
    from tensorflow import keras
    set_seed(seed, threads)
    model = build_model(X.shape[1], X.shape[2], hp)
    cb = [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=FIXED["lr_patience"],
                                            min_delta=0.0001, cooldown=0, min_lr=0)]
    model.fit(X[mask], y[mask], batch_size=hp["batch_size"], epochs=int(epochs), callbacks=cb, verbose=0)
    return model


def compute_mae_days(model, X, y, mask, scale):
    pred = model.predict(X[mask], batch_size=1024, verbose=0)[:, 0]
    return float(np.mean(np.abs(pred - y[mask])) / scale)


# ---------------------------------------------------------------- runs

def list_hyperparameters():
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def read_rows(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    res = pd.read_csv(path)
    return res[res["dataset"] != "dataset"]           # header lines of parallel processes


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def tune_log(ds, args, run_dir):
    """Every setting of the grid (this part of it) with the tuning seed."""
    path = os.path.join(run_dir, "tuning.csv")
    prev = read_rows(path)
    done = set() if prev.empty else set(zip(prev.dataset, prev.units.astype(int), prev.layers.astype(int),
                                              prev.batch_size.astype(int), prev.optimizer))
    hps = list_hyperparameters()[args.part::args.n_parts]
    todo = [hp for hp in hps if (ds, hp["units"], hp["layers"], hp["batch_size"], hp["optimizer"]) not in done]
    print(f"{ds}: {len(todo)} of {len(hps)} settings to tune")
    if not todo:
        return
    log = load_log(ds, args.unit, args.calendar, args.intercase)
    X = build_inputs(log, log["sets"]["train"])
    for hp in todo:
        t0 = time.time()
        val_mae, best_epoch, n_epochs, model = train_with_validation(X, log["y"], log["sets"], hp,
                                                                     args.tune_seed, args.threads)
        row = {"dataset": ds, "variant": args.variant, **hp, "seed": args.tune_seed, "val_mae": val_mae / log["scale"],
               "best_epoch": best_epoch, "epochs_run": n_epochs,
               "test_mae_tuning": compute_mae_days(model, X, log["y"], log["sets"]["test"], log["scale"]),
               "seconds": round(time.time() - t0, 1)}
        append_row(path, row)
        print(f"{ds} {hp} val MAE {row['val_mae']:.3f} (epoch {best_epoch} of {n_epochs}), {row['seconds']} s",
              flush=True)


def refit_log(ds, args, run_dir):
    """Select the setting with the lowest validation MAE and retrain it on
    training + validation cases with the refit seeds."""
    tune = read_rows(os.path.join(run_dir, "tuning.csv"))
    if tune.empty or ds not in set(tune["dataset"]):
        print(f"{ds}: no tuning results yet; no refit")
        return
    tune = tune[tune["dataset"] == ds].astype({"val_mae": float, "best_epoch": float})
    n_grid = len(list_hyperparameters())
    if tune.drop_duplicates(list(GRID)).shape[0] < n_grid:
        print(f"{ds}: tuning not finished ({len(tune)} of {n_grid} settings); no refit")
        return
    best = tune.loc[tune["val_mae"].idxmin()]
    hp = {"units": int(best.units), "layers": int(best.layers), "batch_size": int(best.batch_size),
          "optimizer": best.optimizer}
    sel_path = os.path.join(run_dir, "selected.csv")
    sel = read_rows(sel_path)
    if sel.empty or ds not in set(sel["dataset"]):
        append_row(sel_path, {"dataset": ds, "variant": args.variant, **hp, "val_mae": best.val_mae,
                              "epochs": int(best.best_epoch)})
    path = os.path.join(run_dir, "refit.csv")
    prev = read_rows(path)
    done = set() if prev.empty else set(zip(prev.dataset, prev.seed.astype(int)))
    todo = [s for s in args.seeds if (ds, s) not in done]
    print(f"{ds}: selected {hp}, {int(best.best_epoch)} epochs; {len(todo)} seeds to retrain")
    if not todo:
        return
    log = load_log(ds, args.unit, args.calendar, args.intercase)
    whole = log["sets"]["train"] | log["sets"]["val"]
    X = build_inputs(log, whole)
    for seed in todo:
        t0 = time.time()
        model = train_fixed_epochs(X, log["y"], whole, hp, best.best_epoch, seed, args.threads)
        mae = compute_mae_days(model, X, log["y"], log["sets"]["test"], log["scale"])
        append_row(path, {"dataset": ds, "variant": args.variant, "period": 0, "k": 0, **hp, "seed": seed,
                          "epochs": int(best.best_epoch), "mae": mae})
        print(f"{ds} seed {seed}: test MAE {mae:.3f} days ({time.time() - t0:.0f} s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--refit", action="store_true", help="select on validation and retrain with --seeds")
    parser.add_argument("--tune-seed", type=int, default=14)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12, 13, 14])
    parser.add_argument("--part", type=int, default=0, help="with --n-parts: which part of the grid to tune")
    parser.add_argument("--n-parts", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads of TensorFlow")
    parser.add_argument("--unit", choices=["days", "seconds"], default="days", help="unit of the training label")
    parser.add_argument("--calendar", action="store_true",
                        help="F2 of every event as input, instead of the calendar columns of the benchmark")
    parser.add_argument("--intercase", action="store_true",
                        help="new cases and active resources of every event as input, besides open_cases")
    parser.add_argument("--run-name", default=None, help="default: verenich_lstm, or verenich_lstm_f2 with --calendar")
    args = parser.parse_args()
    args.variant = VARIANT + ("+F2" if args.calendar else "") + ("+IC" if args.intercase else "")
    default = "verenich_lstm" + ("_f2" if args.calendar else "") + ("_intercase" if args.intercase else "")
    run_dir = os.path.join(EVAL_DIR, "results", "runs", args.run_name or default)
    os.makedirs(run_dir, exist_ok=True)
    for ds in args.datasets:
        (refit_log if args.refit else tune_log)(ds, args, run_dir)


if __name__ == "__main__":
    main()