"""Inter-case baselines measured at the events of a prefix: Base-agg (Bagg3)
and Base-index (Bindex).

Both use the three characteristics of B (dyRtp/features/baseline_snapshot.py),
measured at each event of a prefix instead of only at tau: the open cases at
the event, and the arrivals and the active resources in the day before the
event. They differ only in how the values of the events enter the model:

  Bagg3   aggregated over the events of the prefix (mean, max, min, sum,
          std), as the aggregation encoding of Verenich et al. (see
          test_baseline_agg.py); 15 columns
  Bindex  one column per characteristic and event position, as the
          index-based encoding of Verenich et al.; positions after the end
          of the prefix are 0, as the benchmark's fillna; 3 x (maximum
          prefix length) columns

The features are computed once per log and cached in
data/prepared/<log>_bagg3.pkl and <log>_bindex.pkl. hp_tune.py reads them
with load_bagg and trains F1+F2+Bagg3 and F1+F2+Bindex with the tuning
protocol of the other configurations.

Usage (only builds the caches):
    python evaluation/run_bagg_baseline.py
    python evaluation/run_bagg_baseline.py --features Bindex --datasets helpdesk sepsis
"""
import argparse
import os

import numpy as np
import pandas as pd

import experiment_lag as E
from experiment_lag import R, D
from dyRtp import verenich as V
from dyRtp.features.baseline_snapshot import count_recent_arrivals, count_recent_resources

CHARACTERISTICS = ("open_cases", "arrivals", "active_resources")
FEATURES = ["Bagg3", "Bindex"]


def measure_characteristics_at_events(ds, exp_cfg, ds_cfg):
    """Open cases at each event, and arrivals and active resources in the day
    before each event (the definitions of B, measured at the event instead of
    at tau), with the case and the number of the event in its case. Events in
    the order of the benchmark (by case, then time)."""
    ov = exp_cfg["verenich_overrides"]
    dm = V.get_manager(ds, E.RAW_DIR, ov["drop_columns"], ov["rename_columns"])
    data = dm.read_dataset()
    cc, tc, rc = dm.case_id_col, dm.timestamp_col, ds_cfg[ds]["resource_col"]
    ev = data[[cc, tc]].copy()
    ev[tc] = pd.to_datetime(ev[tc])
    ev = ev.sort_values([cc, tc], kind="mergesort")      # event order within a case, as in the benchmark
    t = ev[tc].values.astype("datetime64[ns]")
    ev["open_cases"] = E.build_snapshot_features(data, cc, rc, tc, t, E.INTERVAL)["B__open_cases"].values
    ev["arrivals"] = count_recent_arrivals(data, cc, tc, t, E.INTERVAL)
    ev["active_resources"] = count_recent_resources(data, rc, tc, t, E.INTERVAL)
    ev[D.EVENT_NR] = (ev.groupby(cc).cumcount() + 1).values
    ev[D.CASE] = ev[cc].astype(str).values
    return ev[[D.CASE, D.EVENT_NR] + list(CHARACTERISTICS)].reset_index(drop=True)


def aggregate_over_prefix(ev, case_col, value_col, prefix):
    """Mean, max, min, sum and std of value_col over the first n events of
    each case, for every n (ev sorted by case and time)."""
    v = ev[value_col].astype(float)
    g = v.groupby(ev[case_col])
    n = g.cumcount() + 1
    total = g.cumsum()
    sq = (v ** 2).groupby(ev[case_col]).cumsum()
    var = (sq - total ** 2 / n) / (n - 1).where(n > 1)          # sample variance, as pandas std
    return pd.DataFrame({f"{prefix}{value_col}_mean": total / n, f"{prefix}{value_col}_max": g.cummax(),
                         f"{prefix}{value_col}_min": g.cummin(), f"{prefix}{value_col}_sum": total,
                         f"{prefix}{value_col}_std": np.sqrt(var.clip(lower=0)).fillna(0)})


def compute_aggregated_characteristics(ds, tbl, exp_cfg, ds_cfg):
    """Bagg3: the three characteristics at each event, aggregated over the
    events of each prefix. One row per row of tbl, matched on case and
    prefix length."""
    ev = measure_characteristics_at_events(ds, exp_cfg, ds_cfg)
    agg = pd.concat([aggregate_over_prefix(ev, D.CASE, c, "Bagg3__") for c in CHARACTERISTICS], axis=1)
    agg[D.CASE], agg[D.EVENT_NR] = ev[D.CASE].values, ev[D.EVENT_NR].values
    keys = tbl[[D.CASE, D.EVENT_NR]].astype({D.CASE: str, D.EVENT_NR: int})
    out = keys.merge(agg, on=[D.CASE, D.EVENT_NR], how="left")
    single = (out[D.EVENT_NR] == 1).values
    for c in CHARACTERISTICS:                                    # check: length-1 prefixes equal B at tau
        diff = np.abs(out.loc[single, f"Bagg3__{c}_mean"].values - tbl.loc[single, f"B__{c}"].values)
        print(f"  check {c}: prefixes of length 1 with Bagg3 mean != B: {(diff > 1e-9).sum()} of {single.sum()}")
    return out.drop(columns=[D.CASE, D.EVENT_NR]).set_index(tbl.index)


def compute_indexed_characteristics(ds, tbl, exp_cfg, ds_cfg):
    """Bindex: the three characteristics at each event of the prefix, one
    column per characteristic and event position; positions after the end
    of the prefix are 0, as the benchmark's fillna. One row per row of tbl,
    matched on case and prefix length."""
    ev = measure_characteristics_at_events(ds, exp_cfg, ds_cfg)
    n_max = int(tbl[D.EVENT_NR].max())
    ev = ev[ev[D.EVENT_NR] <= n_max]
    wide = ev.pivot(index=D.CASE, columns=D.EVENT_NR, values=list(CHARACTERISTICS))
    wide.columns = [f"Bindex__{c}_{i}" for c, i in wide.columns]
    cols = [f"Bindex__{c}_{i}" for c in CHARACTERISTICS for i in range(1, n_max + 1)]
    keys = tbl[[D.CASE, D.EVENT_NR]].astype({D.CASE: str, D.EVENT_NR: int})
    out = keys.merge(wide, left_on=D.CASE, right_index=True, how="left").reindex(columns=[D.EVENT_NR] + cols)
    nr = out[D.EVENT_NR].to_numpy()
    vals = out[cols].to_numpy(float)
    pos = np.tile(np.arange(1, n_max + 1), len(CHARACTERISTICS))
    vals[pos[None, :] > nr[:, None]] = 0.0                       # events after the end of the prefix
    for j, c in enumerate(CHARACTERISTICS):                      # check: the value at the last event equals B at tau
        last = vals[np.arange(len(vals)), j * n_max + nr - 1]
        diff = np.abs(last - tbl[f"B__{c}"].to_numpy())
        print(f"  check {c}: prefixes whose value at the last event != B: {(diff > 1e-9).sum()} of {len(diff)}")
    return pd.DataFrame(np.nan_to_num(vals).astype(np.float32), columns=cols, index=tbl.index)


def load_bagg(ds, tbl, exp_cfg, ds_cfg, features):
    """Bagg3 or Bindex features of a log, one row per row of its feature
    table (cached in data/prepared/<log>_<features>.pkl)."""
    path = os.path.join(E.PREP_DIR, f"{ds}_{features.lower()}.pkl")
    if os.path.exists(path):
        cached = pd.read_pickle(path)
        if len(cached) == len(tbl):
            return cached
    if features == "Bindex":
        out = compute_indexed_characteristics(ds, tbl, exp_cfg, ds_cfg)
    else:
        out = compute_aggregated_characteristics(ds, tbl, exp_cfg, ds_cfg)
    out.to_pickle(path)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build the caches of Bagg3 and Bindex.")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=FEATURES)
    parser.add_argument("--datasets", nargs="+", default=None, help="default: all prepared logs, smallest first")
    args = parser.parse_args()
    exp_cfg, ds_cfg = R.load_config("experiment.yaml"), R.load_config("datasets.yaml")
    table = lambda ds: os.path.join(E.PREP_DIR, f"{ds}_table.pkl")
    datasets = args.datasets or sorted((ds for ds in E.list_datasets() if os.path.exists(table(ds))),
                                       key=lambda ds: os.path.getsize(table(ds)))
    for ds in datasets:
        tbl = pd.read_pickle(table(ds))
        for features in args.features:
            E.log(f"{ds}: {features} ...")
            out = load_bagg(ds, tbl, exp_cfg, ds_cfg, features)
            E.log(f"{ds}: {features}, {out.shape[1]} columns")
    E.log("done")


if __name__ == "__main__":
    main()