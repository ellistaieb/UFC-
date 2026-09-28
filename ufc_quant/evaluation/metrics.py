"""Probabilistic scoring and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def score(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    out = {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table with equal-width bins on [0, 1]. Symmetric data -> symmetric table."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    g = df.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).reset_index()
    g["y_se"] = np.sqrt(g["y_rate"] * (1 - g["y_rate"]) / g["n"])
    return g


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float(np.sum(t["n"] * (t["p_mean"] - t["y_rate"]).abs()) / t["n"].sum())
