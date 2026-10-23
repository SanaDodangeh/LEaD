

"""Helper functions.

1. Warm-up and cool-down removal (trim_warmup_cooldown_by_active_cases,
   compute_trimming_window).
2. Description of the daily time series of a log (select_train_period,
   analyze_series_stats, compute_variance_fractions), used to relate the
   gain of F3/F4 to properties of the process dynamics.
"""
import numpy as np
import pandas as pd

from dyRtp.features.F3_F4_process import build_interval_grid, compute_active_cases


def trim_warmup_cooldown_by_active_cases(active_case_series, data_series, q_level=0.2, debug=False):
    """Unchanged from the cbf repository."""
    # Ensure datetime index
    active_case_series = active_case_series.copy()
    data_series = data_series.copy()

    active_case_series.index = pd.to_datetime(active_case_series.index)
    data_series.index = pd.to_datetime(data_series.index)

    active_case_series = active_case_series.sort_index()
    data_series = data_series.sort_index()

    # Align on common timestamps (robust)
    common_idx = active_case_series.index.intersection(data_series.index)
    if common_idx.empty:
        if debug:
            print("[TRIM-DEBUG] No common index between active_case_series and data_series -> no trimming")
            print(f"[TRIM-DEBUG] act_c idx head={list(active_case_series.index[:3])}")
            print(f"[TRIM-DEBUG] data idx head={list(data_series.index[:3])}")
        return data_series

    act = active_case_series.reindex(common_idx)
    y = data_series.reindex(common_idx)

    if debug:
        print(f"[TRIM-DEBUG] min_active_cases=2, q={q_level}")
        print(f"[TRIM-DEBUG] act_c min={act.min()}, max={act.max()}, mean={act.mean()}")
        print(f"[TRIM-DEBUG] common_idx len={len(common_idx)}")

    valid_act = act[act > 2].dropna()
    if valid_act.empty:
        if debug:
            print("[TRIM-DEBUG] valid_act is empty -> no trimming")
        return y  # return aligned y

    cut_value = np.quantile(valid_act.values, q_level)
    in_scope_idx = [i for i, v in enumerate(act.values) if pd.notna(v) and v > cut_value]

    if not in_scope_idx:
        if debug:
            print("[TRIM-DEBUG] in_scope_idx empty -> no trimming")
        return y

    start_i, end_i = in_scope_idx[0], in_scope_idx[-1]
    trimmed = y.iloc[start_i:end_i + 1]
    return trimmed


def compute_trimming_window(data, case_col, ts_col, interval, q_level):
    """Start and end of the period kept after trimming. Prediction points
    with tau in (start, end] are kept, where start is the beginning of the
    first kept interval and end the end of the last kept interval."""
    grid = build_interval_grid(data[ts_col], interval)
    act = compute_active_cases(data, case_col, ts_col, grid)
    kept = trim_warmup_cooldown_by_active_cases(act, act, q_level)
    return kept.index[0] - pd.Timedelta(interval), kept.index[-1]




# description of time series

def select_train_period(series, test_start, q_level=0.2):
    """Part of a daily time series that is available for training: after
    warm-up removal and before the first test case starts. The warm-up is
    found on the active cases series (first entry of the list), with the
    same method as compute_trimming_window."""
    active = series[0]
    kept = trim_warmup_cooldown_by_active_cases(active, active, q_level)
    start = kept.index[0]
    return [s[(s.index >= start) & (s.index < test_start)] for s in series]


def analyze_series_stats(series: pd.Series):
    """
    Computes overall statistics from a time series that is already aggregated per window.

    Parameters:
        series (pd.Series): Time series with datetime index (already window-aggregated)

    Returns:
        dict: {
            'std_dev': float,
            'mean_per_window': float,
            'coefficient_of_variation': float,
            'num_windows': int,
            'sum_total': float
        }
    """
    series = pd.to_numeric(series, errors='coerce').dropna()

    if series.empty:
        return {
            "std_dev": 0,
            "mean_per_window": 0,
            "coefficient_of_variation": 0,
            "num_windows": 0,
            "sum_total": 0
        }
    std_dev = series.std()
    mean_val = series.mean()

    if mean_val != 0:
        cv = std_dev / mean_val
    else:
        cv = None

    return {
        "std_dev": round(std_dev, 5),
        "mean_per_window": round(mean_val, 5),
        "coefficient_of_variation": round(cv, 5) if cv is not None else None,
        "num_windows": len(series),
        "sum_total": round(series.sum(), 5)
    }


def compute_variance_fractions(series: pd.Series, period=7, robust=False):
    """Component variance fractions (BPM 2026 paper): one standard STL fit
    on the whole series, then Var(component) / (Var(T) + Var(S) + Var(R))
    for trend, seasonal and residual. Covariances between the components
    are excluded, so the three fractions sum to 1.

    Returns dict: {'vf_trend', 'vf_seasonal', 'vf_residual'} (NaN if the
    series is too short for two seasonal cycles or constant)."""
    from statsmodels.tsa.seasonal import STL
    y = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    nan = {"vf_trend": np.nan, "vf_seasonal": np.nan, "vf_residual": np.nan}
    if len(y) < 2 * period:
        return nan
    res = STL(y, period=period, robust=robust).fit()
    var_t, var_s, var_r = np.var(res.trend), np.var(res.seasonal), np.var(res.resid)
    total = var_t + var_s + var_r
    if total == 0:
        return nan
    return {"vf_trend": var_t / total, "vf_seasonal": var_s / total, "vf_residual": var_r / total}