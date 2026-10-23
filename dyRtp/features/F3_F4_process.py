"""F3 and F4: process-level time series, their decomposition, and encodings.

Function names follow verb + object.

F3, observed time series (one value per daily interval (t - 1 day, t]):
  build_interval_grid, compute_active_cases, compute_arrival_rate,
  compute_active_resources
Active cases use the last event of each case to decide whether it is still
running, as open_cases in the benchmark of Verenich et al.; B uses the
same definition, so the comparison of B, F3 and F4 is not affected.

F4, decomposed time series:
  decompose_time_series fits standard STL, for every interval t, on the
  time series up to t (expanding window, no look-ahead) and keeps the
  trend, weekly seasonal and residual of the last k intervals of that fit.
  decompose_all_time_series applies it to several time series.

Encodings, shared by F3 and F4:
  encode_lag and encode_decomposed_lag map each prediction point to the
  last interval that is complete at its tau, so no event after tau is
  used. Column names start with "F3__" or "F4__" so that the feature
  groups can be selected by prefix.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- F3 time series

def build_interval_grid(timestamps, interval):
    """End points of equal-length intervals that cover the log.
    The value of a time series at t describes the process at time t."""
    start = timestamps.min().floor(interval)
    end = timestamps.max().ceil(interval)
    return pd.date_range(start + pd.Timedelta(interval), end, freq=interval)


def compute_active_cases(data, case_col, ts_col, grid):
    """Cases started at or before t whose last event is after t
    (the definition of open_cases in the benchmark of Verenich et al.)."""
    g = data.groupby(case_col)[ts_col]
    starts = np.sort(g.min().values)
    ends = np.sort(g.max().values)
    t = grid.values
    started = np.searchsorted(starts, t, side="right")   # start <= t
    ended = np.searchsorted(ends, t, side="right")       # end <= t
    return pd.Series(started - ended, index=grid, name="active_cases")


def compute_arrival_rate(data, case_col, ts_col, grid):
    """Cases whose first event falls in the interval (t - interval, t].
    The boundaries match compute_active_cases: a case that starts exactly
    at t is active at t and has arrived in the interval ending at t."""
    starts = np.sort(data.groupby(case_col)[ts_col].min().values)
    t = grid.values
    step = grid[1] - grid[0]
    upto_t = np.searchsorted(starts, t, side="right")                       # start <= t
    upto_prev = np.searchsorted(starts, (grid - step).values, side="right")  # start <= t - interval
    return pd.Series(upto_t - upto_prev, index=grid, name="arrival_rate")


def find_interval_index(timestamps, grid):
    """Position in grid of the interval (t - interval, t] that contains
    each timestamp."""
    return np.searchsorted(grid.values, pd.DatetimeIndex(timestamps).values, side="left")


def compute_active_resources(data, res_col, ts_col, grid):
    """Distinct resources that execute at least one event in the interval
    (t - interval, t]. Events without a resource are not counted."""
    ev = data[[res_col, ts_col]].dropna(subset=[res_col])
    pos = find_interval_index(ev[ts_col], grid)
    counts = ev.groupby(pos)[res_col].nunique()
    values = counts.reindex(np.arange(len(grid)), fill_value=0).values
    return pd.Series(values, index=grid, name="active_resources")


# ------------------------------------------------------------- F4 components

def decompose_time_series(series, max_k=10, period=7, min_cycles=2, robust=False):
    """Expanding-window STL. For every interval t, standard STL is fitted on
    the series up to t, and the components of the last max_k intervals
    (t, t-1, ..., t-max_k+1) of that fit are kept. A prediction point whose
    last complete interval is t thus gets all its lags from the same fit,
    which uses no interval after t. The first min_cycles * period
    intervals have too little history and are NaN.

    Returns a dict {"<name>_trend": W, "<name>_seasonal": W,
    "<name>_residual": W}, where W is a DataFrame indexed like the series
    with columns lag1 .. lag<max_k> (lag1 = interval t itself)."""
    from statsmodels.tsa.seasonal import STL
    v = series.values.astype(float)
    n = len(v)
    out = {c: np.full((n, max_k), np.nan) for c in ["trend", "seasonal", "residual"]}
    for t in range(min_cycles * period - 1, n):
        res = STL(v[:t + 1], period=period, robust=robust).fit()
        m = min(max_k, t + 1)
        out["trend"][t, :m] = res.trend[::-1][:m]
        out["seasonal"][t, :m] = res.seasonal[::-1][:m]
        out["residual"][t, :m] = res.resid[::-1][:m]
    cols = [f"lag{l+1}" for l in range(max_k)]
    return {f"{series.name}_{c}": pd.DataFrame(W, index=series.index, columns=cols)
            for c, W in out.items()}


def decompose_all_time_series(series_list, max_k=10, period=7):
    """decompose_time_series for several time series, merged into one dict
    with 3 entries per time series."""
    components = {}
    for s in series_list:
        components.update(decompose_time_series(s, max_k=max_k, period=period))
    return components


# ----------------------------------------------------------------- encodings

def get_window_values(series, tau, k):
    """For each prediction time tau, the values of the last k complete
    intervals. Column 0 is the most recent interval (lag 1)."""
    idx = np.searchsorted(series.index.values, tau, side="right") - 1
    v = series.values.astype(float)
    W = np.full((len(tau), k), np.nan)
    for l in range(k):
        j = idx - l
        W[:, l] = np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)
    return W, idx


def encode_lag(series, tau, k, group="F3"):
    """Lag encoding: the last k values, one column per interval."""
    W, _ = get_window_values(series, tau, k)
    return pd.DataFrame(W, columns=[f"{group}__{series.name}_lag{l+1}" for l in range(k)])


def encode_decomposed_lag(name, windows, tau, k, group="F4"):
    """Lag encoding of one F4 component. windows is an entry of
    decompose_time_series; the row of the last complete interval before
    tau is selected and its first k lags are returned."""
    idx = np.searchsorted(windows.index.values, tau, side="right") - 1
    W = windows.values[np.clip(idx, 0, None), :k].copy()
    W[idx < 0] = np.nan
    return pd.DataFrame(W, columns=[f"{group}__{name}_lag{l+1}" for l in range(k)])