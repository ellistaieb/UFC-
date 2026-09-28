"""Model vs market: same fights, same information horizon.

Three distinct questions are answered separately:
  1. probability quality     -> metrics.score on all evaluable fights
  2. value added vs market   -> same-fight comparison, encompassing regression, bootstrap
  3. financial performance   -> backtest module
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ufc_quant.data.odds import pair_two_sided, select_pre_decision_odds
from ufc_quant.models.core import log_loss, logit


def market_probabilities(odds: pd.DataFrame, fights: pd.DataFrame, feats: pd.DataFrame, cfg: dict,
                         bookmaker: str | None = None) -> pd.DataFrame:
    """Pre-decision two-sided odds of one bookmaker, oriented like the features (fighter A)."""
    ocfg = cfg["odds"]
    bookmaker = bookmaker or ocfg["reference_bookmaker"]
    if odds is None or odds.empty:
        return pd.DataFrame(columns=["fight_id", "bookmaker", "odds_a", "odds_b", "overround", "p_market_a",
                                     "snapshot_precision"])
    sel = select_pre_decision_odds(odds[odds["bookmaker"] == bookmaker], fights,
                                   ocfg["decision_hours_before_commence"], ocfg["accept_date_only_snapshots"])
    return pair_two_sided(sel, feats[["fight_id", "fighter_a_id", "fighter_b_id"]])


def encompassing_regression(y, p_market, p_model) -> dict:
    """Logistic regression y ~ b_mkt * logit(p_market) + b_mod * logit(p_model), no intercept.

    b_mod significantly > 0 means the model carries information not contained in the market
    price (on this sample). Standard errors from the inverse observed Fisher information.
    """
    X = np.column_stack([logit(p_market), logit(p_model)])
    y = np.asarray(y, dtype=float)

    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z))

    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (p - y)

    res = minimize(nll, x0=np.array([1.0, 0.0]), jac=grad, method="BFGS")
    b = res.x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = b / se
    # likelihood-ratio test of b_mod = 0 against the market-only model
    res0 = minimize(lambda c: nll(np.array([c[0], 0.0])), x0=np.array([1.0]), method="BFGS")
    lr = 2 * (res0.fun - res.fun)
    from scipy.stats import chi2
    return {
        "n": int(len(y)),
        "beta_market": float(b[0]), "se_market": float(se[0]),
        "beta_model": float(b[1]), "se_model": float(se[1]),
        "z_model": float(z[1]), "p_value_model_two_sided": float(2 * (1 - norm.cdf(abs(z[1])))),
        "lr_stat_model": float(lr), "lr_p_value": float(1 - chi2.cdf(lr, df=1)),
    }


def event_bootstrap_delta(df: pd.DataFrame, col_a: str, col_b: str, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap by event of mean log-loss(col_a) - log-loss(col_b). Negative favours col_a.

    Resampling whole events keeps the within-event dependence; it still assumes events are
    exchangeable over the period (no regime change), which is a limitation.
    """
    y = df["y"].values
    la = -(y * np.log(np.clip(df[col_a], 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - df[col_a], 1e-6, 1)))
    lb = -(y * np.log(np.clip(df[col_b], 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - df[col_b], 1e-6, 1)))
    d = pd.DataFrame({"event_id": df["event_id"].values, "d": np.asarray(la) - np.asarray(lb)})
    g = d.groupby("event_id")["d"].agg(["sum", "size"])
    sums, sizes = g["sum"].values, g["size"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {"delta_mean": float(sums.sum() / sizes.sum()), "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)), "p_delta_ge_0": float(np.mean(boot >= 0)),
            "n_events": int(len(g)), "n_fights": int(sizes.sum())}


def source_consistency(odds: pd.DataFrame, feats: pd.DataFrame) -> dict:
    """Compare de-vigged probabilities across bookmakers/sources on common fights."""
    books = sorted(odds["bookmaker"].unique()) if not odds.empty else []
    out = {}
    orient = feats[["fight_id", "fighter_a_id", "fighter_b_id"]]
    paired = {b: pair_two_sided(odds[odds["bookmaker"] == b], orient).set_index("fight_id") for b in books}
    for i, b1 in enumerate(books):
        for b2 in books[i + 1:]:
            common = paired[b1].index.intersection(paired[b2].index)
            if len(common) < 30:
                continue
            p1, p2 = paired[b1].loc[common, "p_market_a"], paired[b2].loc[common, "p_market_a"]
            out[f"{b1} vs {b2}"] = {
                "n_common_fights": int(len(common)),
                "corr": float(np.corrcoef(p1, p2)[0, 1]),
                "mean_abs_diff": float(np.mean(np.abs(p1 - p2))),
                "share_favourite_disagreement": float(np.mean((p1 > 0.5) != (p2 > 0.5))),
            }
    return out


def market_log_loss(y, p) -> float:
    return log_loss(y, p)
