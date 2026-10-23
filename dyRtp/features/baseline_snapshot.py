"""B: baseline inter-case features, measured at the prediction time tau.

This is how existing work represents the inter-case context (e.g. the
open_cases attribute in the benchmark of Verenich et al., and the
inter-case features of Senderovich et al.): one value per quantity, taken
at tau, without its history. B measures the same three quantities as F3,
so the only difference to F3 is the representation:

  B   one snapshot at tau: open cases at tau; arrivals and active
      resources over the window (tau - window, tau]
  F3  a daily time series, of which the last k complete days are used

B uses events up to tau itself, so it is more recent than F3 (which ends
at the last complete day); the comparison is therefore not in favour of F3.
Column names start with "B__".
"""
import numpy as np
import pandas as pd

PREFIX = "B__"


def count_open_cases(data, case_col, ts_col, tau):
    """Cases started at or before tau whose last event is after tau
    (open_cases in the benchmark of Verenich et al.)."""
    g = data.groupby(case_col)[ts_col]
    starts = np.sort(g.min().values)
    ends = np.sort(g.max().values)
    started = np.searchsorted(starts, tau, side="right")   # start <= tau
    ended = np.searchsorted(ends, tau, side="right")       # end <= tau
    return started - ended


def count_recent_arrivals(data, case_col, ts_col, tau, window):
    """Cases whose first event falls in (tau - window, tau]."""
    starts = np.sort(data.groupby(case_col)[ts_col].min().values)
    lower = tau - np.timedelta64(pd.Timedelta(window))
    return (np.searchsorted(starts, tau, side="right")
            - np.searchsorted(starts, lower, side="right"))


def count_recent_resources(data, res_col, ts_col, tau, window):
    """Distinct resources with at least one event in (tau - window, tau].
    Events without a resource are not counted."""
    ev = data[[res_col, ts_col]].dropna(subset=[res_col])
    lower = tau - np.timedelta64(pd.Timedelta(window))
    count = np.zeros(len(tau), dtype=int)
    for _, times in ev.groupby(res_col)[ts_col]:
        t = np.sort(times.values)
        n_in_window = (np.searchsorted(t, tau, side="right")
                       - np.searchsorted(t, lower, side="right"))
        count += n_in_window > 0
    return count


def build_snapshot_features(data, case_col, res_col, ts_col, tau, window="1D"):
    """The three B features for each prediction time in tau."""
    tau = np.asarray(tau, dtype="datetime64[ns]")
    return pd.DataFrame({
        PREFIX + "open_cases": count_open_cases(data, case_col, ts_col, tau),
        PREFIX + "arrivals": count_recent_arrivals(data, case_col, ts_col, tau, window),
        PREFIX + "active_resources": count_recent_resources(data, res_col, ts_col, tau, window),
    }).astype(float)