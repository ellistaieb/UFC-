"""Win-probability models with enforced A/B symmetry.

Every model returns P(fighter A wins). Symmetry P(A beats B) = 1 - P(B beats A) is enforced by:
  - training on the fight AND its swapped copy (same temporal segment, same fold),
  - averaging at prediction time: p = (f(x) + 1 - f(swap(x))) / 2.
Betting odds are NOT model inputs (they are an external benchmark).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ufc_quant.features.elo import elo_expected
from ufc_quant.features.engine import CONTEXT_FEATURES, DIFF_FEATURES, FIGHTER_FEATURES

EPS = 1e-6


def swap_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """Swap fighters A and B: diff_* negated, a_* <-> b_*, y -> 1 - y. Context unchanged."""
    out = df.copy()
    for c in df.columns:
        if c.startswith("diff_"):
            out[c] = -df[c]
        elif c.startswith("a_") and "b_" + c[2:] in df.columns:
            out[c] = df["b_" + c[2:]]
            out["b_" + c[2:]] = df[c]
    if "fighter_a_id" in df:
        out["fighter_a_id"], out["fighter_b_id"] = df["fighter_b_id"], df["fighter_a_id"]
    if "y" in df:
        out["y"] = 1.0 - df["y"]
    return out


def augment(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, swap_orientation(df)], ignore_index=True)


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class BaseModel:
    name = "base"
    features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "BaseModel":
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba_a(self, df: pd.DataFrame) -> np.ndarray:
        p = 0.5 * (self._raw(df) + 1.0 - self._raw(swap_orientation(df)))
        return np.clip(p, EPS, 1 - EPS)

    def params(self) -> dict:
        return {}


class CoinFlipBaseline(BaseModel):
    """P = 0.5: the natural benchmark once the orientation is random."""

    name = "baseline"

    def _raw(self, df):
        return np.full(len(df), 0.5)


class EloModel(BaseModel):
    name = "elo"
    features = ["a_elo", "b_elo"]

    def _raw(self, df):
        return elo_expected(df["a_elo"].values, df["b_elo"].values)


class LogisticModel(BaseModel):
    """L2 logistic regression on pre-fight differences, no intercept (antisymmetric model)."""

    name = "logistic"
    features = DIFF_FEATURES

    def __init__(self, C: float = 0.1, seed: int = 42):
        self.C = C
        self.seed = seed

    def fit(self, df):
        tr = augment(df)
        # imputer and scaler are fitted on the (augmented) training rows only
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000))
        self.pipe.fit(tr[self.features], tr["y"].astype(int))
        return self

    def _raw(self, df):
        return self.pipe.predict_proba(df[self.features])[:, 1]

    def coefficients(self) -> pd.Series:
        """Coefficients on standardised features (log-odds per 1 SD). Associations, not causal effects."""
        return pd.Series(self.pipe[-1].coef_[0], index=self.features).sort_values(key=np.abs, ascending=False)

    def params(self):
        return {"C": self.C}


class GBMModel(BaseModel):
    """Histogram gradient boosting (sklearn), native NaN handling."""

    name = "gbm"
    features = DIFF_FEATURES + CONTEXT_FEATURES

    def __init__(self, seed: int = 42, **hp):
        self.hp = hp
        self.seed = seed

    def fit(self, df):
        tr = augment(df)
        self.model = HistGradientBoostingClassifier(random_state=self.seed, early_stopping=False, **self.hp)
        self.model.fit(tr[self.features], tr["y"].astype(int))
        return self

    def _raw(self, df):
        return self.model.predict_proba(df[self.features])[:, 1]

    def params(self):
        return dict(self.hp)


class TemperatureCalibrator:
    """p_cal = sigmoid(T * logit(p)). One parameter, preserves symmetry (no intercept).

    Must be fitted only on out-of-sample predictions that precede the evaluated period.
    """

    def __init__(self):
        self.T = 1.0

    def fit(self, p, y) -> "TemperatureCalibrator":
        z = logit(p)
        res = minimize_scalar(lambda t: log_loss(y, sigmoid(t * z)), bounds=(0.05, 5.0), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, p):
        return np.clip(sigmoid(self.T * logit(p)), EPS, 1 - EPS)


def make_model(name: str, params: dict | None = None, seed: int = 42) -> BaseModel:
    params = params or {}
    if name == "baseline":
        return CoinFlipBaseline()
    if name == "elo":
        return EloModel()
    if name == "logistic":
        return LogisticModel(seed=seed, **params)
    if name == "gbm":
        return GBMModel(seed=seed, **params)
    raise ValueError(name)


__all__ = ["FIGHTER_FEATURES", "make_model", "augment", "swap_orientation", "TemperatureCalibrator",
           "log_loss", "logit", "sigmoid"]
