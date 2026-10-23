"""F2: calendar features of the prediction time tau.

All features come from the timestamp of the last event of the prefix.
The month is included only for logs that span more than one year, since
in shorter logs it would act as a proxy for the trend captured by F4.
"""
import pandas as pd

from dyRtp.data import TS

PREFIX = "F2__"


def check_log_spans_year(df):
    """True if the prediction times of the log cover more than one year."""
    return (df[TS].max() - df[TS].min()) > pd.Timedelta(days=365)


def compute_calendar_features(df, include_month):
    """Hour, day of week and, if include_month, month of tau."""
    ts = df[TS]
    out = pd.DataFrame(index=df.index)
    out["hour"] = ts.dt.hour.astype(float)
    out["day_of_week"] = ts.dt.dayofweek.astype(float)
    if include_month:
        out["month"] = ts.dt.month.astype(float)
    out.columns = [PREFIX + c for c in out.columns]
    return out
