"""F1: case features.

The aggregation encoding of the benchmark by Verenich et al., i.e. its
"static" and "agg" encoders (verenich, unchanged):
  - static: case attributes, taken from the first event of the case
  - agg:    per categorical event attribute, the count of each value;
            per numeric event attribute, mean, std, min, max and sum

Added here: time since the previous event, time since the start of the
case, and the number of events in the prefix.
"""
import pandas as pd

from dyRtp.verenich import EncoderFactory

PREFIX = "F1__"


def add_event_columns(dm, data):
    """Per-event helper column, added before prefix generation:
    seconds since the previous event of the same case (0 for the first)."""
    data = data.sort_values([dm.case_id_col, dm.timestamp_col], kind="mergesort")
    g = data.groupby(dm.case_id_col, sort=False)
    data["_time_since_prev"] = (data[dm.timestamp_col] - g[dm.timestamp_col].shift(1)
                                ).dt.total_seconds().fillna(0.0)
    return data


class CaseFeatures:
    """Builds F1 for the prefixes of one part (train, val, test).

    The benchmark encoders fix their columns on the first call, so the
    training prefixes must be transformed first; later calls are aligned
    to these columns."""

    def __init__(self, dm):
        self.dm = dm
        args = dict(case_id_col=dm.case_id_col,
                    static_cat_cols=dm.static_cat_cols, static_num_cols=dm.static_num_cols,
                    dynamic_cat_cols=dm.dynamic_cat_cols, dynamic_num_cols=dm.dynamic_num_cols,
                    fillna=True)
        self.names = ["static", "agg"]
        self.encoders = [EncoderFactory.get_encoder(m, **args) for m in self.names]

    def encode_prefixes(self, prefixes):
        """One row per prefix id, sorted like the benchmark encoders' output."""
        dm = self.dm
        # benchmark encoding, like its FeatureUnion but keeping column names
        parts = [enc.fit(prefixes).transform(prefixes) for enc in self.encoders]
        cols = []
        for name, part in zip(self.names, parts):
            cols += [f"{name}_{c}" for c in part.columns]
        out = pd.concat(parts, axis=1)
        out.columns = cols

        # temporal attributes of the prefix
        g = prefixes.groupby(dm.case_id_col, sort=True)
        tau = g[dm.timestamp_col].max()
        out["time_since_prev"] = g["_time_since_prev"].last()
        out["time_since_start"] = (tau - g[dm.timestamp_col].min()).dt.total_seconds()
        out["event_nr"] = g.size().astype(float)

        out.columns = [PREFIX + c for c in out.columns]
        return out