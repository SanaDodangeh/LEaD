"""Access to the benchmark code of Verenich et al. (verenich,
commit 6075e4d, unchanged) and the experimental protocol built on it.

Used from the benchmark here:
  - dataset_confs.py      column roles of each log
  - DatasetManager        reading the log, temporal split, prefix generation,
                          labels, maximum prefix length

The benchmark's encoders (EncoderFactory) are used for F1, see
dyRtp/features/F1_case.py.

Added here (not part of the benchmark):
  - the overrides of dataset_confs listed in configs/experiment.yaml
  - the prediction time tau, the prefix length and the original case of
    each prefix
"""
import os
import sys

import pandas as pd

VERENICH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "verenich")
if VERENICH_DIR not in sys.path:
    sys.path.insert(0, VERENICH_DIR)

import dataset_confs as vconf  # noqa: E402
import EncoderFactory  # noqa: E402,F401  (re-exported for dyRtp.features.F1_case)
from DatasetManager import DatasetManager  # noqa: E402

COL_KEYS = ["dynamic_cat_cols", "static_cat_cols", "dynamic_num_cols", "static_num_cols"]


def get_manager(dataset, raw_dir, drop_columns, rename_columns):
    """Create the benchmark's DatasetManager, with the data path and the
    documented column overrides applied to its configuration."""
    if dataset not in vconf.filename:
        raise KeyError(f"{dataset} is not defined in verenich/dataset_confs.py")
    vconf.filename[dataset] = os.path.join(raw_dir, os.path.basename(vconf.filename[dataset]))
    for key in COL_KEYS:
        cols = getattr(vconf, key)[dataset]
        cols = [rename_columns.get(c, c) for c in cols]
        getattr(vconf, key)[dataset] = [c for c in cols if c not in drop_columns]
    return DatasetManager(dataset)


def add_case_attributes(dm, data):
    """Per-event helper columns needed after prefix generation, which
    renames the case id ("10" -> "10_2") and overwrites the case length."""
    data = data.sort_values([dm.case_id_col, dm.timestamp_col], kind="mergesort")
    g = data.groupby(dm.case_id_col, sort=False)
    data["_case"] = data[dm.case_id_col]
    data["_orig_case_length"] = g[dm.timestamp_col].transform("size")
    return data


def get_prefix_meta(dm, prefixes):
    """tau, length, original case and original case length of each prefix
    (indexed by prefix id, sorted like the benchmark encoders' output)."""
    g = prefixes.groupby(dm.case_id_col, sort=True)
    return pd.DataFrame({
        "case": g["_case"].first(),
        "tau": g[dm.timestamp_col].max(),
        "event_nr": g.size(),
        "orig_case_length": g["_orig_case_length"].first(),
    })


def get_labels (dm, prefixes):
    """Remaining time of each prefix, by the benchmark's label function.
    Only the columns it reads are passed, since on newer pandas it fails
    on other object columns."""
    return dm.get_label_numeric(prefixes[[dm.case_id_col, dm.label_col]])