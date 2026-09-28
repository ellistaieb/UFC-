"""Chronological splits. Splits are defined on dates, so all fights of an event share a segment."""

from __future__ import annotations

import pandas as pd


def modelling_mask(df: pd.DataFrame) -> pd.Series:
    """Fights usable for supervised learning / evaluation: decided (win/loss), unambiguous ids."""
    return df["y"].notna() & ~df["id_ambiguous"]


def prediction_mask(df: pd.DataFrame) -> pd.Series:
    """Fights we predict (and may bet on): every outcome incl. draws / no contests, since the
    outcome is unknown at decision time. Fights with ambiguous fighter ids are excluded."""
    return ~df["id_ambiguous"]


def walk_forward_folds(df: pd.DataFrame, cfg: dict, include_undecided: bool = False):
    """Yield (year, train_df, valid_df): train on [train_start, Jan 1st of year), predict that year.

    Only dates before `test_start` are ever used here. With include_undecided, valid_df also
    contains draws / no contests (y = NaN) for the backtest.
    """
    s = cfg["splits"]
    test_start = pd.Timestamp(s["test_start"])
    base = df[modelling_mask(df) & (df["event_date"] >= s["train_start"]) & (df["event_date"] < test_start)]
    pred_base = df[prediction_mask(df) & (df["event_date"] < test_start)] if include_undecided else base
    for year in s["walk_forward_years"]:
        start, end = pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year + 1}-01-01"), test_start)
        tr = base[base["event_date"] < start]
        va = pred_base[(pred_base["event_date"] >= start) & (pred_base["event_date"] < end)]
        if len(tr) and len(va):
            yield year, tr, va


def final_split(df: pd.DataFrame, cfg: dict, include_undecided: bool = False):
    """(train_df, test_df): train on everything before test_start, test = most recent period."""
    s = cfg["splits"]
    m = modelling_mask(df)
    tr = df[m & (df["event_date"] >= s["train_start"]) & (df["event_date"] < s["test_start"])]
    mt = prediction_mask(df) if include_undecided else m
    te = df[mt & (df["event_date"] >= s["test_start"])]
    return tr, te


def assert_event_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = set(train["event_id"]) & set(test["event_id"])
    if shared:
        raise AssertionError(f"events present in both segments: {sorted(shared)[:5]}")
    if len(train) and len(test) and train["event_date"].max() >= test["event_date"].min():
        raise AssertionError("training data is not strictly earlier than evaluation data")
