"""Regressors for remaining time.

Tabular models (train_and_predict): XGBoost with early stopping on a
temporal validation split, and a random forest (no early stopping; the
validation split is then used only to select the settings, e.g. the period).

LSTM (train_and_predict_lstm), as a robustness check with a sequence model.
Input, following the LSTM setup of the benchmark by Verenich et al., is the
sequence of the events of the prefix (build_sequence_data). Each event is
described by
  - its categorical attributes (activity, resource, ...), one-hot encoded
    with the most frequent values of the fit cases (the others share one
    slot), case attributes repeated at every event
  - its numeric attributes, standardized on the fit events
  - hours since the previous event and since the start of the case (log1p,
    standardized), hour of day and day of week (sine and cosine)
The inter-case features of a feature set (B, F3, F4 or F4ts, plus the
calendar features F2 of tau) are one vector per prefix
(prepare_extra_features). It is joined with the last hidden state of the
LSTM (late fusion), so that the comparison of B with F3 and F4 is the same
as for the tabular models. Training: Nadam, L1 loss on the standardized
log1p remaining time, dropout, gradient clipping (as in the code of Tax et
al., on which the benchmark's LSTM settings are based), early stopping on
the validation MAE. torch is imported only when the LSTM is used.
"""
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# random forests accept NaN only from scikit-learn 1.4 on; before, NaN is
# replaced by this value (outside the range of the features)
RF_NAN_FILL = -1.0
RF_SUPPORTS_NAN = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 4)


def build_xgb(params, seed):
    """XGBoost regressor; early stopping is set in params."""
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, **params)


def build_rf(params, seed):
    """Random forest regressor with fixed hyperparameters."""
    return RandomForestRegressor(random_state=seed, **params)


def fill_missing_values(X):
    """NaN -> RF_NAN_FILL for scikit-learn versions whose forests reject NaN."""
    X = np.asarray(X, dtype=float)
    return X if RF_SUPPORTS_NAN else np.where(np.isnan(X), RF_NAN_FILL, X)


def train_and_predict(X_fit, y_fit, X_val, y_val, X_test, params, seed):
    params = dict(params)
    name = params.pop("name", "xgb")
    log_target = params.pop("log_target", True)

    t_fit, t_val = y_fit, y_val
    if log_target:
        t_fit, t_val = np.log1p(y_fit), np.log1p(y_val)

    start = time.time()
    if name == "rf":
        model = build_rf(params, seed)
        X_fit, X_val, X_test = (fill_missing_values(X) for X in (X_fit, X_val, X_test))
        model.fit(X_fit, t_fit)
        trees = model.n_estimators
    else:
        model = build_xgb(params, seed)
        model.fit(X_fit, t_fit, eval_set=[(X_val, t_val)], verbose=False)
        trees = int(model.best_iteration)
    train_time = time.time() - start

    start = time.time()
    pred = model.predict(X_test)
    predict_time = time.time() - start
    if log_target:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)

    pred_val = model.predict(X_val)
    if log_target:
        pred_val = np.expm1(pred_val)
    pred_val = np.clip(pred_val, 0, None)

    info = {
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "best_iteration": trees,   # XGBoost: trees after early stopping; RF: n_estimators
        "pred_val": pred_val,      # predictions on the validation set, for model selection
    }
    return pred, model, info


# ------------------------------------------------------------------- LSTM

def encode_categorical(values, fit_rows, max_levels):
    """One-hot encoding with the max_levels most frequent values of the fit
    rows; all other values share the last column."""
    values = values.astype(str)
    levels = values[fit_rows].value_counts().index[:max_levels]
    codes = pd.Categorical(values, categories=levels).codes.astype(np.int64)
    codes[codes < 0] = len(levels)
    out = np.zeros((len(values), len(levels) + 1), dtype=np.float32)
    out[np.arange(len(values)), codes] = 1.0
    return out


def standardize_columns(M, fit_rows):
    """Standardize on the fit rows; NaN -> 0 (the mean)."""
    M = M.astype(np.float64)
    mu = np.nanmean(M[fit_rows], axis=0)
    sd = np.nanstd(M[fit_rows], axis=0)
    mu = np.nan_to_num(mu)
    sd = np.where(np.nan_to_num(sd) > 0, np.nan_to_num(sd), 1.0)
    return np.nan_to_num((M - mu) / sd).astype(np.float32)


def build_event_features(dm, data, fit_cases, max_levels=50):
    """Feature matrix with one row per event (data sorted by case and
    time), and the row of the first event of each case."""
    cc, tc = dm.case_id_col, dm.timestamp_col
    data = data.sort_values([cc, tc], kind="mergesort").reset_index(drop=True)
    fit_rows = data[cc].astype(str).isin(fit_cases).values
    parts = []
    for col in dict.fromkeys(dm.dynamic_cat_cols + dm.static_cat_cols):
        parts.append(encode_categorical(data[col], fit_rows, max_levels))
    num_cols = list(dict.fromkeys(dm.dynamic_num_cols + dm.static_num_cols))
    ts = data[tc]
    first = ts.groupby(data[cc]).transform("min")
    previous = ts.groupby(data[cc]).shift(1).fillna(ts)
    hour = (ts.dt.hour + ts.dt.minute / 60).values
    weekday = ts.dt.dayofweek.values
    numeric = np.column_stack(
        [pd.to_numeric(data[c], errors="coerce").values for c in num_cols]
        + [np.log1p((ts - previous).dt.total_seconds().values / 3600),
           np.log1p((ts - first).dt.total_seconds().values / 3600)])
    parts.append(standardize_columns(numeric, fit_rows))
    parts.append(np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                                  np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)]
                                 ).astype(np.float32))
    events = np.hstack(parts)
    case_ids = data[cc].astype(str).values
    starts = pd.Series(np.arange(len(data))).groupby(case_ids).min()
    return events, starts


def build_sequence_data(dm, data, table, case_col, length_col, set_col, max_levels=50):
    """Event features of all events, and for each prefix of the feature
    table the row of its first event and its length."""
    fit_cases = set(table.loc[table[set_col] == "train", case_col].astype(str))
    events, starts = build_event_features(dm, data, fit_cases, max_levels)
    first_row = starts.reindex(table[case_col].astype(str).values).values
    if np.isnan(first_row.astype(float)).any():
        raise ValueError("prefixes of cases that are not in the event log")
    return {"events": events, "first_row": first_row.astype(np.int64),
            "lengths": table[length_col].to_numpy().astype(np.int64)}


def prepare_extra_features(X, fit):
    """Inter-case and calendar features of the prefixes: clipped to the range
    of the fit rows, standardized on the fit rows, NaN -> 0, plus one
    indicator for prefixes with a missing value (F3/F4 at the start of the
    log, where all of them are missing). The clipping matters for the
    residual and seasonal components, whose later values often lie outside
    the range seen in training: a tree-based model treats them like the
    largest value it has seen, whereas a neural network would extrapolate."""
    X = np.asarray(X, dtype=np.float64)
    missing = np.isnan(X).any(axis=1)
    X = np.clip(X, np.nanmin(X[fit], axis=0), np.nanmax(X[fit], axis=0))
    Z = standardize_columns(X, fit)
    if missing[fit].any():
        Z = np.hstack([Z, missing[:, None].astype(np.float32)])
    return Z


def train_and_predict_lstm(seq, extra, y, fit, val, test, params, seed):
    """Train on the fit prefixes, early stopping on the validation MAE;
    returns the predictions for test (seconds) and an info dict as
    dyRtp.model.train_and_predict does."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(params.get("n_threads", 4))
    device = torch.device("cuda" if torch.cuda.is_available() and params.get("gpu", True) else "cpu")
    if device.type == "cuda" and params.get("gpu_memory_fraction"):
        torch.cuda.set_per_process_memory_fraction(params["gpu_memory_fraction"])

    events = torch.from_numpy(seq["events"]).to(device)
    first_row = torch.from_numpy(seq["first_row"]).to(device)
    lengths = torch.from_numpy(seq["lengths"]).to(device)
    extra_t = torch.from_numpy(extra).to(device)
    t = np.log1p(y)
    mu, sd = t[fit].mean(), t[fit].std()
    target = torch.from_numpy(((t - mu) / sd).astype(np.float32)).to(device)

    class SequenceRegressor(nn.Module):
        def __init__(self, n_in, n_extra, units, layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(n_in, units, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(units + n_extra, units),
                                      nn.ReLU(), nn.Dropout(dropout), nn.Linear(units, 1))

        def forward(self, x, lens, z):
            packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                                       enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            return self.head(torch.cat([h[-1], z], dim=1)).squeeze(1)

    def gather_batch(idx):
        """Padded event sequences of the prefixes idx (a tensor)."""
        lens = lengths[idx]
        steps = torch.arange(int(lens.max()), device=device)
        mask = steps[None, :] < lens[:, None]
        rows = torch.where(mask, first_row[idx][:, None] + steps[None, :], torch.zeros_like(mask, dtype=torch.long))
        return events[rows] * mask[..., None], lens

    model = SequenceRegressor(events.shape[1], extra.shape[1], params["units"], params["layers"],
                              params["dropout"]).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=params["lr_patience"])
    loss_fn = nn.L1Loss()
    fit_idx, val_idx, test_idx = (torch.from_numpy(np.where(m)[0]).to(device) for m in (fit, val, test))

    def predict(idx):
        model.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), 4096):
                j = idx[b:b + 4096]
                x, lens = gather_batch(j)
                out.append(model(x, lens, extra_t[j]).cpu().numpy())
        return np.clip(np.expm1(np.concatenate(out) * sd + mu), 0, None)

    start = time.time()
    best, best_state, wait, epochs = np.inf, None, 0, 0
    for epoch in range(params["max_epochs"]):
        model.train()
        perm = fit_idx[torch.randperm(len(fit_idx), device=device)]
        for b in range(0, len(perm), params["batch_size"]):
            j = perm[b:b + params["batch_size"]]
            x, lens = gather_batch(j)
            opt.zero_grad()
            loss = loss_fn(model(x, lens, extra_t[j]), target[j])
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), params["clip_value"])
            opt.step()
        mae = float(np.mean(np.abs(predict(val_idx) - y[val])))
        scheduler.step(mae)
        epochs = epoch + 1
        if mae < best - 1e-9:
            best, wait = mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= params["patience"]:
                break
    model.load_state_dict(best_state)
    train_time = time.time() - start
    start = time.time()
    pred = predict(test_idx)
    predict_time = time.time() - start
    return pred, model, {"train_time_s": train_time, "predict_time_s": predict_time,
                         "best_iteration": epochs, "pred_val": predict(val_idx)}