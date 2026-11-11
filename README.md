# LEaD: Lag Encoding of Process Dynamics for Remaining Time Prediction

This repository is the supplementary material of the paper *"LEaD: Lag Encoding of Process Dynamics for Remaining Time Prediction"*. It contains the implementation of LEaD and everything needed to reproduce the evaluation of the paper.

**Contents**

1. [The approach](#1-the-approach)
2. [Using LEaD](#2-using-lead)
3. [Reproducing the evaluation](#3-reproducing-the-evaluation)

---

## 1. The approach

Remaining time prediction estimates how long a running case of a business process will take until it completes. Besides the case itself, the remaining time depends on the state of the process as a whole, for example on how many cases are running at the same time. Existing approaches capture this state through *inter-case features*, measured at the moment of prediction, aggregated over the case so far, or recorded at the events of the case.

Processes, however, rarely operate under stable conditions: the number of active cases may rise, fall, or follow a weekly pattern. **LEaD** (Lag Encoding of process Dynamics) encodes these recent *process dynamics* as features for remaining time prediction, **independently of when the events of individual cases occur**.

<p align="center">
  <img src="docs/lead_overview.png" alt="Overview of LEaD within the remaining time prediction pipeline" width="850">
</p>

LEaD takes an event log and outputs, for each prefix of a running case, a set of *process dynamics features*. It works in three steps:

1. **Construct process-level time series.** LEaD divides the time span of the log into consecutive windows of equal length (one day in the paper) and measures process-level characteristics in each window. The paper uses three:
   - **new case count (NC):** cases that start within the window;
   - **active case count (AC):** cases that have started but not completed by the end of the window;
   - **active resource count (AR):** distinct resources that execute at least one event within the window.
2. **Decompose the time series** (LEaD-decomposed only). Each time series is decomposed with STL into a **trend** (T), a **seasonal** component with period *p* (S), and a **residual** (R). For a prefix with prediction timepoint τ, the decomposition uses only the windows up to the last window completed before τ, so no future information is used. It needs at least two seasonal periods (2*p* windows); earlier prefixes receive missing values.
3. **Encode the process dynamics.** For a prefix with prediction timepoint τ (the timestamp of its last event), LEaD encodes the last *k* windows completed before τ, the *lags*:
   - **LEaD-raw** encodes their raw values: *k* features per characteristic;
   - **LEaD-decomposed** encodes their trend, seasonal, and residual components: 3*k* features per characteristic.

The process dynamics features are then combined with the other features of each prefix (for example, case attributes and calendar features) and passed to any predictive model. LEaD is model-agnostic: it provides features, not a model.

Two properties distinguish LEaD from inter-case features:
- **Regular windows:** the history LEaD encodes depends only on the prediction timepoint, not on when the case recorded its events. All prefixes whose prediction timepoint falls into the same window receive the same process dynamics features.
- **Fixed history length:** every prefix receives *k* windows of history, including a prefix with a single event, because the time series describe the whole process, not the case.

**Hyperparameters:** the window length, the number of lags *k*, and, for LEaD-decomposed, the seasonal period *p*. In the paper: daily windows, *k* ∈ {2, 7, 14} and *p* ∈ {7, 30} days, selected per log on a validation set.

## 2. Using LEaD

The implementation of LEaD is in `dyRtp/features/F3_F4_process.py`. It works on any event log given as a pandas `DataFrame` with a case identifier, a timestamp, and (for the active resource count) a resource column.

### 2.1 Installation

```bash
git clone https://github.com/SanaDodangeh/LEaD.git
cd LEaD
conda create -n lead python=3.10
conda activate lead
pip install -r requirements.txt
```

Run your code from the root of the repository, so that the package `dyRtp` is found.

### 2.2 Computing the process dynamics features

```python
import numpy as np
import pandas as pd
from dyRtp.features import F3_F4_process as P

log = pd.read_csv("my_log.csv", parse_dates=["timestamp"])   # one row per event
case, ts, res = "case_id", "timestamp", "resource"             # column names of your log

# Step 1: process-level time series in daily windows
grid = P.build_interval_grid(log[ts], "1D")                    # end points of the windows
series = [P.compute_active_cases(log, case, ts, grid),         # AC
          P.compute_arrival_rate(log, case, ts, grid),         # NC
          P.compute_active_resources(log, res, ts, grid)]      # AR

# Prediction timepoints of your prefixes: the timestamp of the last event of each prefix
tau = prefixes["tau"].values.astype("datetime64[ns]")
k, p = 7, 7                                                    # number of lags, seasonal period

# Step 3 (LEaD-raw): raw values of the last k completed windows
lead_raw = pd.concat([P.encode_lag(s, tau, k) for s in series], axis=1)

# Steps 2 and 3 (LEaD-decomposed): trend, seasonal, and residual of the last k windows
components = P.decompose_all_time_series(series, max_k=k, period=p)
lead_decomposed = pd.concat([P.encode_decomposed_lag(name, W, tau, k)
                             for name, W in components.items()], axis=1)
```

Both results have one row per prediction timepoint, in the order of `tau`, and one column per characteristic and lag, for example `F3__active_cases_lag1` or `F4__active_cases_trend_lag1`. `lag1` is the most recent completed window.

### 2.3 Combining them with other features

Concatenate the process dynamics features with the other features of the same prefixes and train any regression model on the remaining time:

```python
X = pd.concat([other_features.reset_index(drop=True), lead_decomposed], axis=1)
model.fit(X[train], y[train])
```

Prefixes early in the log, with fewer than 2*p* completed windows, have missing values in the LEaD-decomposed features. Use a model that supports missing values (for example XGBoost or the random forest of scikit-learn ≥ 1.4) or remove these prefixes.

### 2.4 Practical notes

- **Choosing *k* and *p*:** select them on a validation set, like the other hyperparameters of the model. The decomposition with period *p* is meaningful only if the time series covers several cycles.
- **Other characteristics:** any process-level characteristic measured per window can be used. Pass a `pandas.Series` indexed by `grid`, with a name, to `encode_lag` and `decompose_all_time_series`. For example, `P.compute_avg_multitasking` measures the mean number of cases each working resource works on.
- **Other window lengths:** pass another frequency to `build_interval_grid`, such as `"12h"` or `"7D"`, and choose *p* in windows accordingly (for example *p* = 14 for a weekly pattern in 12-hour windows).
- **Computation cost:** the decomposition fits one STL per window, so its cost depends on the number of windows, not on the number of prefixes.

## 3. Reproducing the evaluation

The evaluation compares LEaD-raw and LEaD-decomposed with three baselines on 15 real-life event logs, with a random forest and XGBoost. As a deep learning reference, it uses the LSTM of the benchmark of Verenich et al. All steps below are run from the root of the repository.

### 3.1 Environments

The tree-based models, the preparation, and the reports use the environment of Section 2.1. The LSTM needs TensorFlow, with other major versions of NumPy and pandas, so it gets its own environment:

```bash
conda create -n lead-lstm python=3.11
conda activate lead-lstm
pip install -r requirements_lstm.txt
```

| Package | `requirements.txt` (Python 3.10) | `requirements_lstm.txt` (Python 3.11) |
|---|---|---|
| numpy | 1.26.4 | 2.4.6 |
| pandas | 2.3.3 | 3.1.0 |
| scipy | 1.15.2 | |
| statsmodels | 0.15.0 | |
| scikit-learn | 1.7.2 | 1.9.1 |
| xgboost | 3.2.0 | |
| PyYAML | 6.0.3 | 6.0.3 |
| matplotlib | 3.10.9 | |
| tensorflow | | 2.21.0 |

### 3.2 Event logs

We use the preprocessed event logs of the benchmark of Verenich et al. (file `experiments/logdata/logs.zip` of [time-prediction-benchmark](https://github.com/verenich/time-prediction-benchmark)). Download the file and unzip the CSV files into `data/raw/`:

```bash
mkdir -p data/raw
unzip logs.zip -d data/raw/
```

Of the 17 logs of the benchmark, we exclude *credit* (no resource information) and *invoice* (its time span is too short for a reliable decomposition). The remaining 15 logs are:

| Log | Code name | Cases | Events | Time span (days) |
|---|---|---:|---:|---:|
| BPIC11 | `bpic2011` | 1,140 | 149,730 | 1,172 |
| BPIC12-A | `bpic2012a` | 12,007 | 53,952 | 152 |
| BPIC12-O | `bpic2012o` | 3,487 | 15,908 | 165 |
| BPIC12-W | `bpic2012w` | 9,650 | 72,380 | 165 |
| BPIC15-1 | `bpic20151` | 696 | 28,775 | 1,611 |
| BPIC15-2 | `bpic20152` | 753 | 41,202 | 1,704 |
| BPIC15-3 | `bpic20153` | 1,328 | 57,488 | 1,889 |
| BPIC15-4 | `bpic20154` | 577 | 24,234 | 1,930 |
| BPIC15-5 | `bpic20155` | 1,051 | 54,562 | 1,922 |
| BPIC17 | `bpic2017` | 31,509 | 561,671 | 397 |
| Helpdesk | `helpdesk` | 3,218 | 10,598 | 982 |
| Hospital Billing | `hospital` | 59,228 | 330,945 | 1,130 |
| Production | `production` | 225 | 4,543 | 89 |
| Sepsis | `sepsis` | 1,035 | 5,176 | 476 |
| RTFM | `traffic_fines` | 150,370 | 561,470 | 4,917 |

### 3.3 Preparing the logs

```bash
python evaluation/experiment_lag.py --prepare-only --periods 7 30 --ks 14
```

For each log, this writes to `data/prepared/` the prefixes with their set (training, validation, test), remaining time, and other features, the daily time series of AC, NC, and AR, and their decomposition for *p* = 7 and *p* = 30. The settings are in `configs/experiment.yaml`:

| Setting | Value | Meaning |
|---|---|---|
| `split.train_ratio` | 0.8 | Cases ordered by start time; the last 20% form the test set. |
| `split.val_ratio` | 0.2 | The last 20% of the remaining cases form the validation set. |
| `prefixes.max_length_cap`, `max_length_quantile` | 20, 0.9 | Prefixes of length 1 up to the 0.9 quantile of the case lengths of each log, but at most 20 events. |
| `prefixes.drop_complete_traces` | true | Complete traces (remaining time 0) are not used. |
| `trimming.q_level` | 0.2 | Prefixes whose prediction timepoint lies in the warm-up or cool-down period (daily active cases below the 0.2 quantile) are excluded; the windows themselves stay in the time series. |

The other features, the same for all configurations, are the case-level features of the benchmark (`F1`) and the calendar features of τ (`F2`: hour, day of week, and, for logs spanning more than a year, month).

### 3.4 Baseline features

The three baselines represent the same characteristics as LEaD, but as inter-case features of the prefix:

| Baseline | Code name | Encoding |
|---|---|---|
| Base-last | `F1+F2+B` | AC at τ, NC and AR in the 24 hours before τ (computed in Section 3.3). |
| Base-agg | `F1+F2+Bagg3` | The same values at each event of the prefix, aggregated by mean, max, min, sum, and standard deviation. |
| Base-index | `F1+F2+Bindex` | The same values at each event of the prefix, one feature per event position; positions after the end of the prefix are 0. |

LEaD-raw and LEaD-decomposed are `F1+F2+F3` and `F1+F2+F4`. Base-agg and Base-index are computed once per log and cached in `data/prepared/`:

```bash
python evaluation/run_bagg_baseline.py
```

The script checks that the value at the last event of each prefix equals Base-last at τ. Each `check` line it prints should report `0 of ...` prefixes.

### 3.5 Random forest and XGBoost

`evaluation/hp_tune.py` trains and tests both models with all five configurations in three steps:

1. **Hyperparameter selection:** every candidate combination is trained on the training set with one random seed (14) and evaluated on the validation set.
2. **Selection:** per log and configuration, the combination with the lowest validation MAE is selected.
3. **Retraining and testing:** the selected combination is retrained on the training and validation sets with five random seeds (10–14) and evaluated on the test set.

```bash
for m in xgb rf; do
  python evaluation/hp_tune.py --model $m --run-name tune_${m}_s14 --seeds 14 --n-jobs 8
  python evaluation/hp_tune.py --model $m --run-name tune_${m}_s14 --select --seeds 14
  python evaluation/hp_tune.py --model $m --run-name tune_${m}_s14 --refit --seeds 10 11 12 13 14 --n-jobs 8
done
```

| Applies to | Hyperparameter | Candidate values |
|---|---|---|
| XGBoost | grid of the benchmark (`n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `max_depth`) | 48 combinations |
| Random forest | `max_features` | {0.33, 1.0}; 300 trees, `min_samples_leaf` 5 |
| LEaD-raw, LEaD-decomposed | number of lags *k* | {2, 7, 14} |
| LEaD-decomposed | seasonal period *p* | {7, 30} days |

Finished fits are skipped, so a stopped run continues where it stopped. Several processes can work on different logs of the same run in parallel (`--datasets bpic2011 helpdesk ...`). Results are written to `evaluation/results/runs/tune_<model>_s14/`: `all_logs.csv` (tuning), `selected.csv` (selection), and `refit.csv` (test MAE of each retrained model).

### 3.6 LSTM reference

The LSTM of the benchmark (architecture, grid, and training procedure of its `train_LSTM.py`, in `tf.keras`) receives, **at each event of the prefix**, the event attributes of the benchmark, the calendar features (`--calendar`), and the three characteristics AC, NC, and AR (`--intercase`). It uses the same prefixes and sets as the tree-based models. Its grid has 24 combinations (units {100, 200}, layers {1, 2, 3}, batch size {8, 32}, optimizer RMSprop or Nadam), and it can be split over several processes:

```bash
conda activate lead-lstm
for ds in bpic2011 bpic2012a bpic2012o bpic2012w bpic20151 bpic20152 bpic20153 bpic20154 bpic20155 bpic2017 helpdesk hospital production sepsis traffic_fines; do
  for part in 0 1 2 3; do
    python evaluation/run_lstm.py --calendar --intercase --datasets $ds --part $part --n-parts 4 --threads 4 &
  done
  wait
  python evaluation/run_lstm.py --calendar --intercase --datasets $ds --refit --threads 4
done
```

The refit uses the combination with the lowest validation MAE, retrains it for its best number of epochs on the training and validation sets with seeds 10–14, and evaluates it on the test set. Results are written to `evaluation/results/runs/verenich_lstm_f2_intercase/`.

Expect long runtimes: on BPIC17 and RTFM, a single combination with 200 units and batch size 8 can take more than a day on a CPU.

### 3.7 Tables of the paper

```bash
python evaluation/report.py --tables
```

This reads the `refit.csv` files of the three runs and writes the tables of the paper as LaTeX, together with their numbers as CSV, to `evaluation/results/report/tables/`. It trains nothing.

| Paper | File |
|---|---|
| Table 3: mean MAE (with standard deviations) | `tab_mae.tex`, `mae.csv` |
| Table 4: Wilcoxon signed-rank tests | `tab_wilcoxon.tex`, `wilcoxon.csv` |
| Table 5: correlations of the gains with VF(S), and with VF(T) and VF(R) | `tab_corr_gains.tex`, `corr_gains.csv` |
| Variance fractions of all logs | `tab_variance_fractions.tex`, `variance_fractions.csv` |
| Gains of every comparison | `gains.csv` |

How the numbers are computed:
- **MAE:** in days (Sepsis: minutes); mean and standard deviation over the five retrained models.
- **Gain:** reduction of the mean MAE in percent, (MAE<sub>reference</sub> − MAE<sub>configuration</sub>) / MAE<sub>reference</sub> × 100, per log.
- **Statistical significance:** one-sided Wilcoxon signed-rank test over the 15 logs on the gains (alternative: the gain is positive), significance level 0.05; effect size: rank-biserial correlation.
- **Variance fractions:** VF(X) = Var(X) / (Var(T) + Var(S) + Var(R)) of the STL components (*p* = 7) of each characteristic, on the windows before the test period only.
- **Correlations:** Pearson correlation over the 15 logs between the gains of a comparison and the VFs of a characteristic.

### 3.8 Main results

- At least one LEaD configuration is more accurate than all three baselines on **10 of 15 logs with the random forest** and **8 of 15 with XGBoost**. The largest gains are on BPIC11 and Production, where LEaD-decomposed is more than 36% and 15% more accurate than all three baselines with both models.
- With the random forest, LEaD-decomposed is significantly more accurate than all three baselines, and LEaD-raw than Base-agg, with effect sizes between 0.57 and 0.83.
- The gains of LEaD-decomposed are positively correlated with the seasonal variance fractions of NC and AR (0.33–0.49): encoding process dynamics is particularly beneficial when new cases and active resources follow a weekly pattern.

The data preparation, the encodings of the other features, the baselines' encoding strategies, and the LSTM build on the benchmark of Verenich et al.: I. Verenich, M. Dumas, M. La Rosa, F. M. Maggi, and I. Teinemaa, *Survey and cross-benchmark comparison of remaining time prediction methods in business process monitoring*, ACM TIST 10(4), 2019.