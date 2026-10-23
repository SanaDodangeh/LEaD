"""Standard column names of the feature table.

The feature table has one row per prefix (prediction point). Its meta
columns are listed in META_COLS; every other column is a feature whose
name starts with its group ("F1__", "F2__", ...). F3 and F4 are joined
on TS, the prediction time tau.
"""
PREFIX_ID = "prefix_id"   # benchmark id: "<case>" for length 1, "<case>_<n>" otherwise
CASE = "case_id"
EVENT_NR = "event_nr"     # number of events in the prefix
TS = "tau"                # timestamp of the last event of the prefix
TARGET = "remtime"        # remaining time in seconds (benchmark label)
SET = "set"               # train, val or test

META_COLS = [PREFIX_ID, CASE, EVENT_NR, TS, TARGET, SET]
